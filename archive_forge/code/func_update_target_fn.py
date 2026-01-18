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
@make_tf_callable(self.get_session())
def update_target_fn(tau):
    tau = tf.convert_to_tensor(tau, dtype=tf.float32)
    update_target_expr = []
    assert len(model_vars) == len(target_model_vars), (model_vars, target_model_vars)
    for var, var_target in zip(model_vars, target_model_vars):
        update_target_expr.append(var_target.assign(tau * var + (1.0 - tau) * var_target))
        logger.debug('Update target op {}'.format(var_target))
    return tf.group(*update_target_expr)