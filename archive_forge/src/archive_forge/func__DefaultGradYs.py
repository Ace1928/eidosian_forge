import collections
import contextlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_state
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops, gradient_uid='__unsupported__'):
    """Fill in default values for grad_ys.

  Args:
    grad_ys: List of gradients, can contain None.
    ys: List of tensors.
    colocate_gradients_with_ops: If True, try colocating gradients with
      the corresponding op.
    gradient_uid: A unique identifier within the graph indicating
      which invocation of gradients is being executed. Used to cluster
      ops for compilation.

  Returns:
    A list of gradients to use, without None.

  Raises:
    ValueError: If sizes of gradients and inputs don't match
    TypeError: If type of any gradient is not valid for its input.
  """
    if len(grad_ys) != len(ys):
        raise ValueError(f'Length mismatch. Passed {len(grad_ys)} grad_ys for {len(ys)} ys')
    grad_ys = indexed_slices.convert_n_to_tensor_or_indexed_slices(grad_ys, name='grad_y')
    new_grad_ys = []
    for i, (y, grad_y) in enumerate(zip(ys, grad_ys)):
        with _maybe_colocate_with(y.op, gradient_uid, colocate_gradients_with_ops):
            if grad_y is None:
                if y.dtype.is_complex:
                    raise TypeError(f'Gradients of complex tensors ({y}) must set grad_ys (y.dtype = {dtypes.as_dtype(y.dtype).name})')
                new_grad_ys.append(array_ops.ones(array_ops.shape(y), dtype=y.dtype, name='grad_ys_%d' % i))
                continue
            if y.dtype.is_floating or y.dtype.is_integer:
                if not grad_y.dtype.is_floating and (not grad_y.dtype.is_integer):
                    raise TypeError(f'Gradient type {dtypes.as_dtype(grad_y.dtype).name} generated for real or integer-valued tensor {y} with type {dtypes.as_dtype(y.dtype).name} must be real or integer')
            elif y.dtype.is_complex:
                if not grad_y.dtype.is_complex:
                    raise TypeError(f'Gradient type {dtypes.as_dtype(grad_y.dtype).name} generated for complex-valued tensor {y} with type {dtypes.as_dtype(y.dtype).name} must be real')
            elif y.dtype == dtypes.variant:
                if grad_y.dtype != dtypes.variant:
                    raise TypeError(f'Gradient type {dtypes.as_dtype(grad_y.dtype).name} generated for variant tensor {y} with type {dtypes.as_dtype(y.dtype).name} must be variant')
            elif y.dtype == dtypes.resource:
                if grad_y.dtype == dtypes.resource:
                    raise TypeError(f'Input gradient {grad_y} for resource tensor {y} should not be a resource')
            else:
                raise TypeError(f'Tensor {y} with type {dtypes.as_dtype(y.dtype).name} must be numeric to obtain a default gradient')
            if isinstance(grad_y, indexed_slices.IndexedSlices):
                new_grad_ys.append(indexed_slices.IndexedSlices(indices=array_ops.identity(grad_y.indices, name='grad_ys_%d_indices' % i) if isinstance(grad_y.indices, tensor_lib.Tensor) else grad_y.indices, values=array_ops.identity(grad_y.values, name='grad_ys_%d_values' % i) if isinstance(grad_y.values, tensor_lib.Tensor) else grad_y.values, dense_shape=array_ops.identity(grad_y.dense_shape, name='grad_ys_%d_shape' % i) if isinstance(grad_y.dense_shape, tensor_lib.Tensor) else grad_y.dense_shape))
            else:
                new_grad_ys.append(array_ops.identity(grad_y, name='grad_ys_%d' % i))
    return new_grad_ys