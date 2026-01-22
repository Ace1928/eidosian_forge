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
class KLCoeffMixin:
    """Assigns the `update_kl()` and other KL-related methods to a TFPolicy.

    This is used in Algorithms to update the KL coefficient after each
    learning step based on `config.kl_target` and the measured KL value
    (from the train_batch).
    """

    def __init__(self, config: AlgorithmConfigDict):
        self.kl_coeff_val = config['kl_coeff']
        self.kl_coeff = get_variable(float(self.kl_coeff_val), tf_name='kl_coeff', trainable=False, framework=config['framework'])
        self.kl_target = config['kl_target']
        if self.framework == 'tf':
            self._kl_coeff_placeholder = tf1.placeholder(dtype=tf.float32, name='kl_coeff')
            self._kl_coeff_update = self.kl_coeff.assign(self._kl_coeff_placeholder, read_value=False)

    def update_kl(self, sampled_kl):
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff_val *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff_val *= 0.5
        else:
            return self.kl_coeff_val
        self._set_kl_coeff(self.kl_coeff_val)
        return self.kl_coeff_val

    def _set_kl_coeff(self, new_kl_coeff):
        self.kl_coeff_val = new_kl_coeff
        if self.framework == 'tf':
            self.get_session().run(self._kl_coeff_update, feed_dict={self._kl_coeff_placeholder: self.kl_coeff_val})
        else:
            self.kl_coeff.assign(self.kl_coeff_val, read_value=False)

    @override(Policy)
    def get_state(self) -> PolicyState:
        state = super().get_state()
        state['current_kl_coeff'] = self.kl_coeff_val
        return state

    @override(Policy)
    def set_state(self, state: PolicyState) -> None:
        self._set_kl_coeff(state.pop('current_kl_coeff', self.config['kl_coeff']))
        super().set_state(state)