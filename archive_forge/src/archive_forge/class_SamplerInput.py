import logging
import queue
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
from typing import (
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.env.base_env import ASYNC_RESET_RETURN, BaseEnv, convert_to_base_env
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.env_runner_v2 import (
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.offline import InputReader
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import deprecation_warning, DEPRECATED_VALUE
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.numpy import convert_to_numpy, make_action_immutable
from ray.rllib.utils.spaces.space_utils import clip_action, unbatch, unsquash_action
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
@DeveloperAPI
class SamplerInput(InputReader, metaclass=ABCMeta):
    """Reads input experiences from an existing sampler."""

    @override(InputReader)
    def next(self) -> SampleBatchType:
        batches = [self.get_data()]
        batches.extend(self.get_extra_batches())
        if len(batches) == 0:
            raise RuntimeError('No data available from sampler.')
        return concat_samples(batches)

    @abstractmethod
    @DeveloperAPI
    def get_data(self) -> SampleBatchType:
        """Called by `self.next()` to return the next batch of data.

        Override this in child classes.

        Returns:
            The next batch of data.
        """
        raise NotImplementedError

    @abstractmethod
    @DeveloperAPI
    def get_metrics(self) -> List[RolloutMetrics]:
        """Returns list of episode metrics since the last call to this method.

        The list will contain one RolloutMetrics object per completed episode.

        Returns:
            List of RolloutMetrics objects, one per completed episode since
            the last call to this method.
        """
        raise NotImplementedError

    @abstractmethod
    @DeveloperAPI
    def get_extra_batches(self) -> List[SampleBatchType]:
        """Returns list of extra batches since the last call to this method.

        The list will contain all SampleBatches or
        MultiAgentBatches that the user has provided thus-far. Users can
        add these "extra batches" to an episode by calling the episode's
        `add_extra_batch([SampleBatchType])` method. This can be done from
        inside an overridden `Policy.compute_actions_from_input_dict(...,
        episodes)` or from a custom callback's `on_episode_[start|step|end]()`
        methods.

        Returns:
            List of SamplesBatches or MultiAgentBatches provided thus-far by
            the user since the last call to this method.
        """
        raise NotImplementedError