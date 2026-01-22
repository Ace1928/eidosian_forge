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
def _extra_action_out_impl(self) -> Dict[str, TensorType]:
    extra_action_out = super().extra_action_out_fn()
    if isinstance(self.model, tf.keras.Model):
        return extra_action_out
    extra_action_out.update({SampleBatch.VF_PREDS: self.model.value_function()})
    return extra_action_out