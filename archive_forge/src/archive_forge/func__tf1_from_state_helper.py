import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
import ray
import ray.experimental.tf_utils
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy, PolicyState, PolicySpec
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.error import ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.tf_run_builder import _TFRunBuilder
from ray.rllib.utils.tf_utils import get_gpu_devices
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
@staticmethod
def _tf1_from_state_helper(state: PolicyState) -> 'Policy':
    """Recovers a TFPolicy from a state object.

        The `state` of an instantiated TFPolicy can be retrieved by calling its
        `get_state` method. Is meant to be used by the Policy.from_state() method to
        aid with tracking variable creation.

        Args:
            state: The state to recover a new TFPolicy instance from.

        Returns:
            A new TFPolicy instance.
        """
    serialized_pol_spec: Optional[dict] = state.get('policy_spec')
    if serialized_pol_spec is None:
        raise ValueError('No `policy_spec` key was found in given `state`! Cannot create new Policy.')
    pol_spec = PolicySpec.deserialize(serialized_pol_spec)
    with tf1.variable_scope(TFPolicy.next_tf_var_scope_name()):
        new_policy = pol_spec.policy_class(pol_spec.observation_space, pol_spec.action_space, pol_spec.config)
    new_policy.set_state(state)
    return new_policy