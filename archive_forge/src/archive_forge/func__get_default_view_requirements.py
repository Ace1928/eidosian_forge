import json
import logging
import os
import platform
from abc import ABCMeta, abstractmethod
from typing import (
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Box
from packaging import version
import ray
import ray.cloudpickle as pickle
from ray.actor import ActorHandle
from ray.train import Checkpoint
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import (
from ray.rllib.utils.checkpoints import (
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.serialization import (
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.tensor_dtype import get_np_dtype
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def _get_default_view_requirements(self):
    """Returns a default ViewRequirements dict.

        Note: This is the base/maximum requirement dict, from which later
        some requirements will be subtracted again automatically to streamline
        data collection, batch creation, and data transfer.

        Returns:
            ViewReqDict: The default view requirements dict.
        """
    return {SampleBatch.OBS: ViewRequirement(space=self.observation_space), SampleBatch.NEXT_OBS: ViewRequirement(data_col=SampleBatch.OBS, shift=1, space=self.observation_space, used_for_compute_actions=False), SampleBatch.ACTIONS: ViewRequirement(space=self.action_space, used_for_compute_actions=False), SampleBatch.PREV_ACTIONS: ViewRequirement(data_col=SampleBatch.ACTIONS, shift=-1, space=self.action_space), SampleBatch.REWARDS: ViewRequirement(), SampleBatch.PREV_REWARDS: ViewRequirement(data_col=SampleBatch.REWARDS, shift=-1), SampleBatch.TERMINATEDS: ViewRequirement(), SampleBatch.TRUNCATEDS: ViewRequirement(), SampleBatch.INFOS: ViewRequirement(used_for_compute_actions=False), SampleBatch.EPS_ID: ViewRequirement(), SampleBatch.UNROLL_ID: ViewRequirement(), SampleBatch.AGENT_INDEX: ViewRequirement(), SampleBatch.T: ViewRequirement()}