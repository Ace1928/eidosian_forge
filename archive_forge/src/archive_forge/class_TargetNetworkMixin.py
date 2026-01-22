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
class TargetNetworkMixin:
    """Assign the `update_target` method to the policy.

    The function is called every `target_network_update_freq` steps by the
    master learner.
    """

    def __init__(self):
        if not self.config.get('_enable_new_api_stack', False):
            model_vars = self.model.trainable_variables()
            target_model_vars = self.target_model.trainable_variables()

            @make_tf_callable(self.get_session())
            def update_target_fn(tau):
                tau = tf.convert_to_tensor(tau, dtype=tf.float32)
                update_target_expr = []
                assert len(model_vars) == len(target_model_vars), (model_vars, target_model_vars)
                for var, var_target in zip(model_vars, target_model_vars):
                    update_target_expr.append(var_target.assign(tau * var + (1.0 - tau) * var_target))
                    logger.debug('Update target op {}'.format(var_target))
                return tf.group(*update_target_expr)
            self._do_update = update_target_fn
            self.update_target(tau=1.0)

    @property
    def q_func_vars(self):
        if not hasattr(self, '_q_func_vars'):
            if self.config.get('_enable_new_api_stack', False):
                self._q_func_vars = self.model.variables
            else:
                self._q_func_vars = self.model.variables()
        return self._q_func_vars

    @property
    def target_q_func_vars(self):
        if not hasattr(self, '_target_q_func_vars'):
            if self.config.get('_enable_new_api_stack', False):
                self._target_q_func_vars = self.target_model.variables
            else:
                self._target_q_func_vars = self.target_model.variables()
        return self._target_q_func_vars

    def update_target(self, tau: int=None) -> None:
        self._do_update(np.float32(tau or self.config.get('tau', 1.0)))

    @override(TFPolicy)
    def variables(self) -> List[TensorType]:
        if self.config.get('_enable_new_api_stack', False):
            return self.model.variables
        else:
            return self.model.variables()

    def set_weights(self, weights):
        if isinstance(self, TFPolicy):
            TFPolicy.set_weights(self, weights)
        elif isinstance(self, EagerTFPolicyV2):
            EagerTFPolicyV2.set_weights(self, weights)
        elif isinstance(self, EagerTFPolicy):
            EagerTFPolicy.set_weights(self, weights)
        if not self.config.get('_enable_new_api_stack', False):
            self.update_target(self.config.get('tau', 1.0))