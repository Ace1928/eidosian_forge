import gymnasium as gym
import numpy as np
import tree
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.core.rl_module.rl_module import RLModule, SingleAgentRLModuleSpec
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.utils import _gym_env_creator
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.annotations import ExperimentalAPI, override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import TensorStructType, TensorType
from ray.tune.registry import ENV_CREATOR, _global_registry
@ExperimentalAPI
class SingleAgentEnvRunner(EnvRunner):
    """The generic environment runner for the single agent case."""

    @override(EnvRunner)
    def __init__(self, config: 'AlgorithmConfig', **kwargs):
        super().__init__(config=config)
        self.worker_index: int = kwargs.get('worker_index')
        if isinstance(self.config.env, str) and _global_registry.contains(ENV_CREATOR, self.config.env):
            entry_point = partial(_global_registry.get(ENV_CREATOR, self.config.env), self.config.env_config)
        else:
            entry_point = partial(_gym_env_creator, env_context=self.config.env_config, env_descriptor=self.config.env)
        gym.register('rllib-single-agent-env-runner-v0', entry_point=entry_point)
        self.env: gym.Wrapper = gym.wrappers.VectorListInfo(gym.vector.make('rllib-single-agent-env-runner-v0', num_envs=self.config.num_envs_per_worker, asynchronous=self.config.remote_worker_envs))
        self.num_envs: int = self.env.num_envs
        assert self.num_envs == self.config.num_envs_per_worker
        try:
            module_spec: SingleAgentRLModuleSpec = self.config.get_default_rl_module_spec()
            module_spec.observation_space = self.env.envs[0].observation_space
            module_spec.action_space = self.env.envs[0].action_space
            module_spec.model_config_dict = self.config.model
            self.module: RLModule = module_spec.build()
        except NotImplementedError:
            self.module = None
        self._needs_initial_reset: bool = True
        self._episodes: List[Optional['SingleAgentEpisode']] = [None for _ in range(self.num_envs)]
        self._done_episodes_for_metrics: List['SingleAgentEpisode'] = []
        self._ongoing_episodes_for_metrics: Dict[List] = defaultdict(list)
        self._ts_since_last_metrics: int = 0
        self._weights_seq_no: int = 0
        self._states = [None for _ in range(self.num_envs)]

    @override(EnvRunner)
    def sample(self, *, num_timesteps: int=None, num_episodes: int=None, explore: bool=True, random_actions: bool=False, with_render_data: bool=False) -> List['SingleAgentEpisode']:
        """Runs and returns a sample (n timesteps or m episodes) on the env(s)."""
        assert not (num_timesteps is not None and num_episodes is not None)
        if num_timesteps is None and num_episodes is None:
            if self.config.batch_mode == 'truncate_episodes':
                num_timesteps = self.config.get_rollout_fragment_length(worker_index=self.worker_index) * self.num_envs
            else:
                num_episodes = self.num_envs
        if num_timesteps is not None:
            return self._sample_timesteps(num_timesteps=num_timesteps, explore=explore, random_actions=random_actions, force_reset=False)
        else:
            return self._sample_episodes(num_episodes=num_episodes, explore=explore, random_actions=random_actions, with_render_data=with_render_data)

    def _sample_timesteps(self, num_timesteps: int, explore: bool=True, random_actions: bool=False, force_reset: bool=False) -> List['SingleAgentEpisode']:
        """Helper method to sample n timesteps."""
        from ray.rllib.env.single_agent_episode import SingleAgentEpisode
        done_episodes_to_return: List['SingleAgentEpisode'] = []
        if hasattr(self.module, 'get_initial_state'):
            initial_states = tree.map_structure(lambda s: np.repeat(s, self.num_envs, axis=0), self.module.get_initial_state())
        else:
            initial_states = {}
        if force_reset or self._needs_initial_reset:
            obs, infos = self.env.reset()
            self._needs_initial_reset = False
            self._episodes = [SingleAgentEpisode() for _ in range(self.num_envs)]
            states = initial_states
            for i in range(self.num_envs):
                self._episodes[i].add_env_reset(observation=obs[i], infos=infos[i])
                self._states[i] = {k: s[i] for k, s in states.items()}
        else:
            obs = np.stack([eps.observations[-1] for eps in self._episodes])
            states = {k: np.stack([initial_states[k][i] if state is None else state[k] for i, state in enumerate(self._states)]) for k in initial_states.keys()}
        ts = 0
        while ts < num_timesteps:
            if random_actions:
                actions = self.env.action_space.sample()
                action_logp = np.zeros(shape=(actions.shape[0],))
                fwd_out = {}
            else:
                batch = {STATE_IN: tree.map_structure(lambda s: self._convert_from_numpy(s), states), SampleBatch.OBS: self._convert_from_numpy(obs)}
                from ray.rllib.utils.nested_dict import NestedDict
                batch = NestedDict(batch)
                if explore:
                    fwd_out = self.module.forward_exploration(batch)
                else:
                    fwd_out = self.module.forward_inference(batch)
                actions, action_logp = self._sample_actions_if_necessary(fwd_out, explore)
                fwd_out = convert_to_numpy(fwd_out)
                if STATE_OUT in fwd_out:
                    states = fwd_out[STATE_OUT]
            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            ts += self.num_envs
            for i in range(self.num_envs):
                s = {k: s[i] for k, s in states.items()}
                extra_model_output = {}
                for k, v in fwd_out.items():
                    if SampleBatch.ACTIONS != k:
                        extra_model_output[k] = v[i]
                extra_model_output[SampleBatch.ACTION_LOGP] = action_logp[i]
                if terminateds[i] or truncateds[i]:
                    self._episodes[i].add_env_step(infos[i]['final_observation'], actions[i], rewards[i], infos=infos[i]['final_info'], terminated=terminateds[i], truncated=truncateds[i], extra_model_outputs=extra_model_output)
                    self._states[i] = s
                    if hasattr(self.module, 'get_initial_state'):
                        for k, v in self.module.get_initial_state().items():
                            states[k][i] = convert_to_numpy(v)
                    done_episodes_to_return.append(self._episodes[i].finalize())
                    self._episodes[i] = SingleAgentEpisode(observations=[obs[i]], infos=[infos[i]])
                    self._states[i] = s
                else:
                    self._episodes[i].add_env_step(obs[i], actions[i], rewards[i], infos=infos[i], extra_model_outputs=extra_model_output)
                    self._states[i] = s
        self._done_episodes_for_metrics.extend(done_episodes_to_return)
        new_episodes = [eps.cut() for eps in self._episodes]
        ongoing_episodes_to_return = [episode.finalize() for episode in self._episodes if episode.t > 0]
        for eps in ongoing_episodes_to_return:
            self._ongoing_episodes_for_metrics[eps.id_].append(eps)
        self._ts_since_last_metrics += ts
        self._episodes = new_episodes
        return done_episodes_to_return + ongoing_episodes_to_return

    def _sample_episodes(self, num_episodes: int, explore: bool=True, random_actions: bool=False, with_render_data: bool=False) -> List['SingleAgentEpisode']:
        """Helper method to run n episodes.

        See docstring of `self.sample()` for more details.
        """
        from ray.rllib.env.single_agent_episode import SingleAgentEpisode
        self._needs_initial_reset = True
        done_episodes_to_return: List['SingleAgentEpisode'] = []
        obs, infos = self.env.reset()
        episodes = [SingleAgentEpisode() for _ in range(self.num_envs)]
        if hasattr(self.module, 'get_initial_state'):
            states = tree.map_structure(lambda s: np.repeat(s, self.num_envs, axis=0), self.module.get_initial_state())
        else:
            states = {}
        render_images = [None] * self.num_envs
        if with_render_data:
            render_images = [e.render() for e in self.env.envs]
        for i in range(self.num_envs):
            episodes[i].add_env_reset(observation=obs[i], infos=infos[i], render_image=render_images[i])
        eps = 0
        while eps < num_episodes:
            if random_actions:
                actions = self.env.action_space.sample()
                action_logp = np.zeros(shape=actions.shape[0])
                fwd_out = {}
            else:
                batch = {STATE_IN: tree.map_structure(lambda s: self._convert_from_numpy(s), states), SampleBatch.OBS: self._convert_from_numpy(obs)}
                if explore:
                    fwd_out = self.module.forward_exploration(batch)
                else:
                    fwd_out = self.module.forward_inference(batch)
                actions, action_logp = self._sample_actions_if_necessary(fwd_out, explore)
                fwd_out = convert_to_numpy(fwd_out)
                if STATE_OUT in fwd_out:
                    states = convert_to_numpy(fwd_out[STATE_OUT])
            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            if with_render_data:
                render_images = [e.render() for e in self.env.envs]
            for i in range(self.num_envs):
                extra_model_output = {}
                for k, v in fwd_out.items():
                    if SampleBatch.ACTIONS not in k:
                        extra_model_output[k] = v[i]
                extra_model_output[SampleBatch.ACTION_LOGP] = action_logp[i]
                if terminateds[i] or truncateds[i]:
                    eps += 1
                    episodes[i].add_env_step(infos[i]['final_observation'], actions[i], rewards[i], infos=infos[i]['final_info'], terminated=terminateds[i], truncated=truncateds[i], extra_model_outputs=extra_model_output)
                    done_episodes_to_return.append(episodes[i])
                    if eps == num_episodes:
                        break
                    if hasattr(self.module, 'get_initial_state'):
                        for k, v in self.module.get_initial_state().items():
                            states[k][i] = (convert_to_numpy(v),)
                    episodes[i] = SingleAgentEpisode(observations=[obs[i]], infos=[infos[i]], render_images=None if render_images[i] is None else [render_images[i]])
                else:
                    episodes[i].add_env_step(obs[i], actions[i], rewards[i], infos=infos[i], render_image=render_images[i], extra_model_outputs=extra_model_output)
        self._done_episodes_for_metrics.extend(done_episodes_to_return)
        self._ts_since_last_metrics += sum((len(eps) for eps in done_episodes_to_return))
        return [episode for episode in done_episodes_to_return if episode.t > 0]

    def get_metrics(self) -> List[RolloutMetrics]:
        metrics = []
        for eps in self._done_episodes_for_metrics:
            assert eps.is_done
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

    def set_weights(self, weights, global_vars=None, weights_seq_no: int=0):
        """Writes the weights of our (single-agent) RLModule."""
        if isinstance(weights, dict) and DEFAULT_POLICY_ID in weights:
            weights = weights[DEFAULT_POLICY_ID]
        weights = self._convert_to_tensor(weights)
        self.module.set_state(weights)

    def get_weights(self, modules=None):
        """Returns the weights of our (single-agent) RLModule."""
        return self.module.get_state()

    @override(EnvRunner)
    def assert_healthy(self):
        assert self.env and self.module

    @override(EnvRunner)
    def stop(self):
        self.env.close()

    def _sample_actions_if_necessary(self, fwd_out: TensorStructType, explore: bool=True) -> Tuple[np.array, np.array]:
        """Samples actions from action distribution if necessary."""
        if SampleBatch.ACTIONS in fwd_out.keys():
            actions = convert_to_numpy(fwd_out[SampleBatch.ACTIONS])
            action_logp = convert_to_numpy(fwd_out[SampleBatch.ACTION_LOGP])
        else:
            if explore:
                action_dist_cls = self.module.get_exploration_action_dist_cls()
            else:
                action_dist_cls = self.module.get_inference_action_dist_cls()
            action_dist = action_dist_cls.from_logits(fwd_out[SampleBatch.ACTION_DIST_INPUTS])
            actions = action_dist.sample()
            action_logp = convert_to_numpy(action_dist.logp(actions))
            actions = convert_to_numpy(actions)
        return (actions, action_logp)

    def _convert_from_numpy(self, array: np.array) -> TensorType:
        """Converts a numpy array to a framework-specific tensor."""
        if self.config.framework_str == 'torch':
            return torch.from_numpy(array)
        else:
            return tf.convert_to_tensor(array)

    def _convert_to_tensor(self, struct) -> TensorType:
        """Converts structs to a framework-specific tensor."""
        if self.config.framework_str == 'torch':
            return convert_to_torch_tensor(struct)
        else:
            return tree.map_structure(tf.convert_to_tensor, struct)