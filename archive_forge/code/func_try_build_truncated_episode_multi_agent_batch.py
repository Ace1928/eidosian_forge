import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EpisodeID, PolicyID, TensorType
@abstractmethod
def try_build_truncated_episode_multi_agent_batch(self) -> List[Union[MultiAgentBatch, SampleBatch]]:
    """Tries to build an MA-batch, if `rollout_fragment_length` is reached.

        Any unprocessed data will be first postprocessed with a policy
        postprocessor.
        This is usually called to collect samples for policy training.
        If not enough data has been collected yet (`rollout_fragment_length`),
        returns an empty list.

        Returns:
            List[Union[MultiAgentBatch, SampleBatch]]: Returns a (possibly
                empty) list of MultiAgentBatches (containing the accumulated
                SampleBatches for each policy or a simple SampleBatch if only
                one policy). The list will be empty if
                `self.rollout_fragment_length` has not been reached yet.
        """
    raise NotImplementedError