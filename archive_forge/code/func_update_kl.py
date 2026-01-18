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
def update_kl(self, sampled_kl):
    if sampled_kl > 2.0 * self.kl_target:
        self.kl_coeff_val *= 1.5
    elif sampled_kl < 0.5 * self.kl_target:
        self.kl_coeff_val *= 0.5
    else:
        return self.kl_coeff_val
    self._set_kl_coeff(self.kl_coeff_val)
    return self.kl_coeff_val