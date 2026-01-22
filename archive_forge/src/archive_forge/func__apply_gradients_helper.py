import functools
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple, Union
import tree  # pip install dm_tree
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import add_mixins, force_list
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.error import ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.tf_utils import get_gpu_devices
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def _apply_gradients_helper(self, grads_and_vars):
    self._re_trace_counter += 1
    if apply_gradients_fn:
        if self.config['_tf_policy_handles_more_than_one_loss']:
            apply_gradients_fn(self, self._optimizers, grads_and_vars)
        else:
            apply_gradients_fn(self, self._optimizer, grads_and_vars)
    elif self.config['_tf_policy_handles_more_than_one_loss']:
        for i, o in enumerate(self._optimizers):
            o.apply_gradients([(g, v) for g, v in grads_and_vars[i] if g is not None])
    else:
        self._optimizer.apply_gradients([(g, v) for g, v in grads_and_vars if g is not None])