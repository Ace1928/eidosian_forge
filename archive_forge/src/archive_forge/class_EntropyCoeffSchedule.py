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
class EntropyCoeffSchedule:
    """Mixin for TFPolicy that adds entropy coeff decay."""

    def __init__(self, entropy_coeff, entropy_coeff_schedule):
        self._entropy_coeff_schedule = None
        if entropy_coeff_schedule is None or self.config.get('_enable_new_api_stack', False):
            self.entropy_coeff = get_variable(entropy_coeff, framework='tf', tf_name='entropy_coeff', trainable=False)
        else:
            if isinstance(entropy_coeff_schedule, list):
                self._entropy_coeff_schedule = PiecewiseSchedule(entropy_coeff_schedule, outside_value=entropy_coeff_schedule[-1][-1], framework=None)
            else:
                self._entropy_coeff_schedule = PiecewiseSchedule([[0, entropy_coeff], [entropy_coeff_schedule, 0.0]], outside_value=0.0, framework=None)
            self.entropy_coeff = get_variable(self._entropy_coeff_schedule.value(0), framework='tf', tf_name='entropy_coeff', trainable=False)
            if self.framework == 'tf':
                self._entropy_coeff_placeholder = tf1.placeholder(dtype=tf.float32, name='entropy_coeff')
                self._entropy_coeff_update = self.entropy_coeff.assign(self._entropy_coeff_placeholder, read_value=False)

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._entropy_coeff_schedule is not None:
            new_val = self._entropy_coeff_schedule.value(global_vars['timestep'])
            if self.framework == 'tf':
                self.get_session().run(self._entropy_coeff_update, feed_dict={self._entropy_coeff_placeholder: new_val})
            else:
                self.entropy_coeff.assign(new_val, read_value=False)