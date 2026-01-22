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
def _traced_eager_policy(eager_policy_cls):
    """Wrapper class that enables tracing for all eager policy methods.

    This is enabled by the `--trace`/`eager_tracing=True` config when
    framework=tf2.
    """

    class TracedEagerPolicy(eager_policy_cls):

        def __init__(self, *args, **kwargs):
            self._traced_learn_on_batch_helper = False
            self._traced_compute_actions_helper = False
            self._traced_compute_gradients_helper = False
            self._traced_apply_gradients_helper = False
            super(TracedEagerPolicy, self).__init__(*args, **kwargs)

        @_check_too_many_retraces
        @override(Policy)
        def compute_actions_from_input_dict(self, input_dict: Dict[str, TensorType], explore: bool=None, timestep: Optional[int]=None, episodes: Optional[List[Episode]]=None, **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
            """Traced version of Policy.compute_actions_from_input_dict."""
            if self._traced_compute_actions_helper is False and (not self._no_tracing):
                if self.config.get('_enable_new_api_stack'):
                    self._compute_actions_helper_rl_module_explore = _convert_eager_inputs(tf.function(super(TracedEagerPolicy, self)._compute_actions_helper_rl_module_explore, autograph=True, reduce_retracing=True))
                    self._compute_actions_helper_rl_module_inference = _convert_eager_inputs(tf.function(super(TracedEagerPolicy, self)._compute_actions_helper_rl_module_inference, autograph=True, reduce_retracing=True))
                else:
                    self._compute_actions_helper = _convert_eager_inputs(tf.function(super(TracedEagerPolicy, self)._compute_actions_helper, autograph=False, reduce_retracing=True))
                self._traced_compute_actions_helper = True
            return super(TracedEagerPolicy, self).compute_actions_from_input_dict(input_dict=input_dict, explore=explore, timestep=timestep, episodes=episodes, **kwargs)

        @_check_too_many_retraces
        @override(eager_policy_cls)
        def learn_on_batch(self, samples):
            """Traced version of Policy.learn_on_batch."""
            if self._traced_learn_on_batch_helper is False and (not self._no_tracing):
                self._learn_on_batch_helper = _convert_eager_inputs(tf.function(super(TracedEagerPolicy, self)._learn_on_batch_helper, autograph=False, reduce_retracing=True))
                self._traced_learn_on_batch_helper = True
            return super(TracedEagerPolicy, self).learn_on_batch(samples)

        @_check_too_many_retraces
        @override(eager_policy_cls)
        def compute_gradients(self, samples: SampleBatch) -> ModelGradients:
            """Traced version of Policy.compute_gradients."""
            if self._traced_compute_gradients_helper is False and (not self._no_tracing):
                self._compute_gradients_helper = _convert_eager_inputs(tf.function(super(TracedEagerPolicy, self)._compute_gradients_helper, autograph=False, reduce_retracing=True))
                self._traced_compute_gradients_helper = True
            return super(TracedEagerPolicy, self).compute_gradients(samples)

        @_check_too_many_retraces
        @override(Policy)
        def apply_gradients(self, grads: ModelGradients) -> None:
            """Traced version of Policy.apply_gradients."""
            if self._traced_apply_gradients_helper is False and (not self._no_tracing):
                self._apply_gradients_helper = _convert_eager_inputs(tf.function(super(TracedEagerPolicy, self)._apply_gradients_helper, autograph=False, reduce_retracing=True))
                self._traced_apply_gradients_helper = True
            return super(TracedEagerPolicy, self).apply_gradients(grads)

        @classmethod
        def with_tracing(cls):
            return cls
    TracedEagerPolicy.__name__ = eager_policy_cls.__name__ + '_traced'
    TracedEagerPolicy.__qualname__ = eager_policy_cls.__qualname__ + '_traced'
    return TracedEagerPolicy