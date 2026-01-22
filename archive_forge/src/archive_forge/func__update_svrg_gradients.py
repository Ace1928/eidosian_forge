import time
import logging
import mxnet as mx
from mxnet.module import Module
from .svrg_optimizer import _SVRGOptimizer
def _update_svrg_gradients(self):
    """Calculates gradients based on the SVRG update rule.
        """
    param_names = self._exec_group.param_names
    for ctx in range(self._ctx_len):
        for index, name in enumerate(param_names):
            g_curr_batch_reg = self._exec_group.grad_arrays[index][ctx]
            g_curr_batch_special = self._mod_aux._exec_group.grad_arrays[index][ctx]
            g_special_weight_all_batch = self._param_dict[ctx][name]
            g_svrg = self._svrg_grads_update_rule(g_curr_batch_reg, g_curr_batch_special, g_special_weight_all_batch)
            self._exec_group.grad_arrays[index][ctx] = g_svrg