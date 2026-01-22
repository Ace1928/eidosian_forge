from collections import defaultdict
from functools import partial
from typing import List, Tuple
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.wrappers.atari_wrappers import NoopResetEnv, MaxAndSkipEnv
from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv
from ray.rllib.env.utils import _gym_env_creator
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.numpy import one_hot
from ray.tune.registry import ENV_CREATOR, _global_registry
class DreamerV3EnvRunner(EnvRunner):
    """An environment runner to collect data from vectorized gymnasium environments."""

    def __init__(self, config: AlgorithmConfig, **kwargs):
        """Initializes a DreamerV3EnvRunner instance.

        Args:
            config: The config to use to setup this EnvRunner.
        """
        super().__init__(config=config)
        if self.config.env.startswith('ALE/'):
            from supersuit.generic_wrappers import resize_v1
            wrappers = [partial(gym.wrappers.TimeLimit, max_episode_steps=108000), partial(resize_v1, x_size=64, y_size=64), NormalizedImageEnv, NoopResetEnv, MaxAndSkipEnv]
            self.env = gym.vector.make('GymV26Environment-v0', env_id=self.config.env, wrappers=wrappers, num_envs=self.config.num_envs_per_worker, asynchronous=self.config.remote_worker_envs, make_kwargs=dict(self.config.env_config, **{'render_mode': 'rgb_array'}))
        elif self.config.env.startswith('DMC/'):
            parts = self.config.env.split('/')
            assert len(parts) == 3, f"ERROR: DMC env must be formatted as 'DMC/[task]/[domain]', e.g. 'DMC/cartpole/swingup'! You provided '{self.config.env}'."
            gym.register('dmc_env-v0', lambda from_pixels=True: DMCEnv(parts[1], parts[2], from_pixels=from_pixels, channels_first=False))
            self.env = gym.vector.make('dmc_env-v0', wrappers=[ActionClip], num_envs=self.config.num_envs_per_worker, asynchronous=self.config.remote_worker_envs, **dict(self.config.env_config))
        else:
            gym.register('dreamerv3-custom-env-v0', partial(_global_registry.get(ENV_CREATOR, self.config.env), self.config.env_config) if _global_registry.contains(ENV_CREATOR, self.config.env) else partial(_gym_env_creator, env_context=self.config.env_config, env_descriptor=self.config.env))
            self.env = gym.vector.make('dreamerv3-custom-env-v0', num_envs=self.config.num_envs_per_worker, asynchronous=False)
        self.num_envs = self.env.num_envs
        assert self.num_envs == self.config.num_envs_per_worker
        if self.config.share_module_between_env_runner_and_learner:
            self.module = None
        else:
            policy_dict, _ = self.config.get_multi_agent_setup(env=self.env)
            module_spec = self.config.get_marl_module_spec(policy_dict=policy_dict)
            self.module = module_spec.build()[DEFAULT_POLICY_ID]
        self._needs_initial_reset = True
        self._episodes = [None for _ in range(self.num_envs)]
        self._states = [None for _ in range(self.num_envs)]
        self._done_episodes_for_metrics = []
        self._ongoing_episodes_for_metrics = defaultdict(list)
        self._ts_since_last_metrics = 0

    @override(EnvRunner)
    def sample(self, *, num_timesteps: int=None, num_episodes: int=None, explore: bool=True, random_actions: bool=False, with_render_data: bool=False) -> Tuple[List[SingleAgentEpisode], List[SingleAgentEpisode]]:
        """Runs and returns a sample (n timesteps or m episodes) on the environment(s).

        Timesteps or episodes are counted in total (across all vectorized
        sub-environments). For example, if self.num_envs=2 and num_timesteps=10, each
        sub-environment will be sampled for 5 steps. If self.num_envs=3 and
        num_episodes=30, each sub-environment will be sampled for 10 episodes.

        Args:
            num_timesteps: The number of timesteps to sample from the environment(s).
                Note that only exactly one of `num_timesteps` or `num_episodes` must be
                provided.
            num_episodes: The number of full episodes to sample from the environment(s).
                Note that only exactly one of `num_timesteps` or `num_episodes` must be
                provided.
            explore: Indicates whether to utilize exploration when picking actions.
            random_actions: Whether to only use random actions. If True, the value of
                `explore` is ignored.
            force_reset: Whether to reset the environment(s) before starting to sample.
                If False, will still reset the environment(s) if they were left in
                a terminated or truncated state during previous sample calls.
            with_render_data: If True, will record rendering images per timestep
                in the returned Episodes. This data can be used to create video
                reports.
                TODO (sven): Note that this is only supported for runnign with
                 `num_episodes` yet.

        Returns:
            A tuple consisting of a) list of Episode instances that are done and
            b) list of Episode instances that are still ongoing.
        """
        if num_timesteps is None and num_episodes is None:
            if self.config.batch_mode == 'truncate_episodes':
                num_timesteps = self.config.rollout_fragment_length * self.num_envs
            else:
                num_episodes = self.num_envs
        if num_timesteps is not None:
            return self._sample_timesteps(num_timesteps=num_timesteps, explore=explore, random_actions=random_actions, force_reset=False)
        else:
            return (self._sample_episodes(num_episodes=num_episodes, explore=explore, random_actions=random_actions, with_render_data=with_render_data), [])

    def _sample_timesteps(self, num_timesteps: int, explore: bool=True, random_actions: bool=False, force_reset: bool=False) -> Tuple[List[SingleAgentEpisode], List[SingleAgentEpisode]]:
        """Helper method to run n timesteps.

        See docstring of self.sample() for more details.
        """
        done_episodes_to_return = []
        initial_states = tree.map_structure(lambda s: np.repeat(s, self.num_envs, axis=0), self.module.get_initial_state())
        if force_reset or self._needs_initial_reset:
            obs, _ = self.env.reset()
            self._episodes = [SingleAgentEpisode() for _ in range(self.num_envs)]
            states = initial_states
            is_first = np.ones((self.num_envs,))
            self._needs_initial_reset = False
            for i in range(self.num_envs):
                self._episodes[i].add_env_reset(observation=obs[i])
                self._states[i] = {k: s[i] for k, s in states.items()}
        else:
            obs = np.stack([eps.observations[-1] for eps in self._episodes])
            states = {k: np.stack([initial_states[k][i] if self._states[i] is None else self._states[i][k] for i, eps in enumerate(self._episodes)]) for k in initial_states.keys()}
            is_first = np.zeros((self.num_envs,))
            for i, eps in enumerate(self._episodes):
                if len(eps) == 0:
                    is_first[i] = 1.0
        ts = 0
        while ts < num_timesteps:
            if random_actions:
                actions = self.env.action_space.sample()
            else:
                batch = {STATE_IN: tree.map_structure(lambda s: tf.convert_to_tensor(s), states), SampleBatch.OBS: tf.convert_to_tensor(obs), 'is_first': tf.convert_to_tensor(is_first)}
                if explore:
                    outs = self.module.forward_exploration(batch)
                else:
                    outs = self.module.forward_inference(batch)
                actions = outs[SampleBatch.ACTIONS].numpy()
                if isinstance(self.env.single_action_space, gym.spaces.Discrete):
                    actions = np.argmax(actions, axis=-1)
                states = tree.map_structure(lambda s: s.numpy(), outs[STATE_OUT])
            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            ts += self.num_envs
            for i in range(self.num_envs):
                s = {k: s[i] for k, s in states.items()}
                if terminateds[i] or truncateds[i]:
                    self._episodes[i].add_env_step(observation=infos['final_observation'][i], action=actions[i], reward=rewards[i], terminated=terminateds[i], truncated=truncateds[i])
                    self._states[i] = s
                    for k, v in self.module.get_initial_state().items():
                        states[k][i] = v.numpy()
                    is_first[i] = True
                    done_episodes_to_return.append(self._episodes[i])
                    self._episodes[i] = SingleAgentEpisode(observations=[obs[i]])
                else:
                    self._episodes[i].add_env_step(observation=obs[i], action=actions[i], reward=rewards[i])
                    is_first[i] = False
                self._states[i] = s
        self._done_episodes_for_metrics.extend(done_episodes_to_return)
        ongoing_episodes = self._episodes
        self._episodes = [eps.cut() for eps in self._episodes]
        for eps in ongoing_episodes:
            self._ongoing_episodes_for_metrics[eps.id_].append(eps)
        self._ts_since_last_metrics += ts
        return (done_episodes_to_return, ongoing_episodes)

    def _sample_episodes(self, num_episodes: int, explore: bool=True, random_actions: bool=False, with_render_data: bool=False) -> List[SingleAgentEpisode]:
        """Helper method to run n episodes.

        See docstring of `self.sample()` for more details.
        """
        done_episodes_to_return = []
        obs, _ = self.env.reset()
        episodes = [SingleAgentEpisode() for _ in range(self.num_envs)]
        states = tree.map_structure(lambda s: np.repeat(s, self.num_envs, axis=0), self.module.get_initial_state())
        is_first = np.ones((self.num_envs,))
        render_images = [None] * self.num_envs
        if with_render_data:
            render_images = [e.render() for e in self.env.envs]
        for i in range(self.num_envs):
            episodes[i].add_env_reset(observation=obs[i], render_image=render_images[i])
        eps = 0
        while eps < num_episodes:
            if random_actions:
                actions = self.env.action_space.sample()
            else:
                batch = {STATE_IN: tree.map_structure(lambda s: tf.convert_to_tensor(s), states), SampleBatch.OBS: tf.convert_to_tensor(obs), 'is_first': tf.convert_to_tensor(is_first)}
                if explore:
                    outs = self.module.forward_exploration(batch)
                else:
                    outs = self.module.forward_inference(batch)
                actions = outs[SampleBatch.ACTIONS].numpy()
                if isinstance(self.env.single_action_space, gym.spaces.Discrete):
                    actions = np.argmax(actions, axis=-1)
                states = tree.map_structure(lambda s: s.numpy(), outs[STATE_OUT])
            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            if with_render_data:
                render_images = [e.render() for e in self.env.envs]
            for i in range(self.num_envs):
                if terminateds[i] or truncateds[i]:
                    eps += 1
                    episodes[i].add_env_step(observation=infos['final_observation'][i], action=actions[i], reward=rewards[i], terminated=terminateds[i], truncated=truncateds[i])
                    done_episodes_to_return.append(episodes[i])
                    if eps == num_episodes:
                        break
                    for k, v in self.module.get_initial_state().items():
                        states[k][i] = v.numpy()
                    is_first[i] = True
                    episodes[i] = SingleAgentEpisode(observations=[obs[i]], render_images=[render_images[i]] if with_render_data else None)
                else:
                    episodes[i].add_env_step(observation=obs[i], action=actions[i], reward=rewards[i], render_image=render_images[i])
                    is_first[i] = False
        self._done_episodes_for_metrics.extend(done_episodes_to_return)
        self._ts_since_last_metrics += sum((len(eps) for eps in done_episodes_to_return))
        self._needs_initial_reset = True
        return done_episodes_to_return

    def get_metrics(self) -> List[RolloutMetrics]:
        metrics = []
        for eps in self._done_episodes_for_metrics:
            episode_length = len(eps)
            episode_reward = eps.get_return()
            if eps.id_ in self._ongoing_episodes_for_metrics:
                for eps2 in self._ongoing_episodes_for_metrics[eps.id_]:
                    episode_length += len(eps2)
                    episode_reward += eps2.get_return()
                del self._ongoing_episodes_for_metrics[eps.id_]
            metrics.append(RolloutMetrics(episode_length=episode_length, episode_reward=episode_reward))
        self._done_episodes_for_metrics.clear()
        self._ts_since_last_metrics = 0
        return metrics

    def set_weights(self, weights, global_vars=None):
        """Writes the weights of our (single-agent) RLModule."""
        if self.module is None:
            assert self.config.share_module_between_env_runner_and_learner
        else:
            self.module.set_state(weights[DEFAULT_POLICY_ID])

    @override(EnvRunner)
    def assert_healthy(self):
        assert self.env and self.module

    @override(EnvRunner)
    def stop(self):
        self.env.close()