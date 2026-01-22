from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export
def _apply_sparse_duplicate_indices(self, grad, var):
    delta = indexed_slices.IndexedSlices(grad.values * math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype), grad.indices, grad.dense_shape)
    return var.scatter_sub(delta, use_locking=self._use_locking)