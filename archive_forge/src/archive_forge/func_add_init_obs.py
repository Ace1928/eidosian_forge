import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EpisodeID, PolicyID, TensorType
@abstractmethod
def add_init_obs(self, *, episode: Episode, agent_id: AgentID, policy_id: PolicyID, init_obs: TensorType, init_infos: Optional[Dict[str, TensorType]]=None, t: int=-1) -> None:
    """Adds an initial obs (after reset) to this collector.

        Since the very first observation in an environment is collected w/o
        additional data (w/o actions, w/o reward) after env.reset() is called,
        this method initializes a new trajectory for a given agent.
        `add_init_obs()` has to be called first for each agent/episode-ID
        combination. After this, only `add_action_reward_next_obs()` must be
        called for that same agent/episode-pair.

        Args:
            episode: The Episode, for which we
                are adding an Agent's initial observation.
            agent_id: Unique id for the agent we are adding
                values for.
            env_id: The environment index (in a vectorized setup).
            policy_id: Unique id for policy controlling the agent.
            init_obs: Initial observation (after env.reset()).
            init_obs: Initial observation (after env.reset()).
            init_infos: Initial infos dict (after env.reset()).
            t: The time step (episode length - 1). The initial obs has
                ts=-1(!), then an action/reward/next-obs at t=0, etc..

        .. testcode::
            :skipif: True

            obs, infos = env.reset()
            collector.add_init_obs(
                episode=my_episode,
                agent_id=0,
                policy_id="pol0",
                t=-1,
                init_obs=obs,
                init_infos=infos,
            )
            obs, r, terminated, truncated, info = env.step(action)
            collector.add_action_reward_next_obs(12345, 0, "pol0", False, {
                "action": action, "obs": obs, "reward": r, "terminated": terminated,
                "truncated": truncated, "info": info
            })
        """
    raise NotImplementedError