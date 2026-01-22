from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import smart_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _scale_loss(self, loss):
    loss_scale = self._loss_scale()
    if callable(loss):

        def new_loss():
            loss_val = loss()
            return loss_val * math_ops.cast(loss_scale, loss_val.dtype)
        return new_loss
    else:
        return loss * math_ops.cast(loss_scale, loss.dtype)