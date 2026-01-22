import logging
from typing import Dict, List
import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.eager_tf_policy import EagerTFPolicy
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.framework import get_variable, try_import_tf
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.rllib.utils.tf_utils import make_tf_callable
from ray.rllib.utils.typing import (
@DeveloperAPI
class ValueNetworkMixin:
    """Assigns the `_value()` method to a TFPolicy.

    This way, Policy can call `_value()` to get the current VF estimate on a
    single(!) observation (as done in `postprocess_trajectory_fn`).
    Note: When doing this, an actual forward pass is being performed.
    This is different from only calling `model.value_function()`, where
    the result of the most recent forward pass is being used to return an
    already calculated tensor.
    """

    def __init__(self, config):
        if config.get('use_gae') or config.get('vtrace'):

            @make_tf_callable(self.get_session())
            def value(**input_dict):
                input_dict = SampleBatch(input_dict)
                if isinstance(self.model, tf.keras.Model):
                    _, _, extra_outs = self.model(input_dict)
                    return extra_outs[SampleBatch.VF_PREDS][0]
                else:
                    model_out, _ = self.model(input_dict)
                    return self.model.value_function()[0]
        else:

            @make_tf_callable(self.get_session())
            def value(*args, **kwargs):
                return tf.constant(0.0)
        self._value = value
        self._should_cache_extra_action = config['framework'] == 'tf'
        self._cached_extra_action_fetches = None

    def _extra_action_out_impl(self) -> Dict[str, TensorType]:
        extra_action_out = super().extra_action_out_fn()
        if isinstance(self.model, tf.keras.Model):
            return extra_action_out
        extra_action_out.update({SampleBatch.VF_PREDS: self.model.value_function()})
        return extra_action_out

    def extra_action_out_fn(self) -> Dict[str, TensorType]:
        if not self._should_cache_extra_action:
            return self._extra_action_out_impl()
        if self._cached_extra_action_fetches is not None:
            return self._cached_extra_action_fetches
        self._cached_extra_action_fetches = self._extra_action_out_impl()
        return self._cached_extra_action_fetches