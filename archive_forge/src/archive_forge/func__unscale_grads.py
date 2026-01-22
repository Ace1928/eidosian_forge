from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import smart_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _unscale_grads(self, grads):
    loss_scale = self._loss_scale()
    loss_scale_reciprocal = 1 / loss_scale
    return [None if g is None else self._scale_grad(g, loss_scale_reciprocal) for g in grads]