from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export
def _finish(self, update_ops, name_scope):
    with ops.control_dependencies(update_ops):
        beta1_power, beta2_power = self._get_beta_accumulators()
        with ops.colocate_with(beta1_power):
            update_beta1 = beta1_power.assign(beta1_power * self._beta1_t, use_locking=self._use_locking)
            update_beta2 = beta2_power.assign(beta2_power * self._beta2_t, use_locking=self._use_locking)
    return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name=name_scope)