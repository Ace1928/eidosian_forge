from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import smart_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _scale_grad(self, grad, loss_scale_reciprocal):
    if isinstance(grad, indexed_slices.IndexedSlices):
        grad_vals = grad.values * loss_scale_reciprocal
        return indexed_slices.IndexedSlices(grad_vals, grad.indices, grad.dense_shape)
    return grad * loss_scale_reciprocal