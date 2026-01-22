import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def _compute_theoretical_jacobian(f, y_shape, y_dtype, xs, param):
    """Computes the theoretical Jacobian for f regarding xs[param].

  One can think of the relation among f, xs and y as y = f(xs).

  Args:
    f: the function.
    y_shape: the shape of the result.
    y_dtype: the dtype of the result.
    xs: a list of tensors.
    param: the index of the target parameter.

  Returns:
    A 2-d numpy array representing the Jacobian. It has "y_size" rows
    and "x_size" columns where "x_size" is the number of elements in xs[param]
    and "y_size" is the number of elements in the result.

  Raises:
    ValueError: If result is empty but the gradient is nonzero.
  """
    x = xs[param]
    x_shape = tuple(x.shape) + (2,) if x.dtype.is_complex else x.shape
    y_factor = 2 if y_dtype.is_complex else 1
    x_size = _product(x_shape)
    x_val_size = _product(x_shape[1:])
    y_size = _product(y_shape) * y_factor
    jacobian = np.zeros((y_size, x_size), dtype=x.dtype.real_dtype.as_numpy_dtype)
    dy_data = np.zeros(y_shape, dtype=y_dtype.as_numpy_dtype)
    dy_data_flat = dy_data.ravel().view(y_dtype.real_dtype.as_numpy_dtype)
    grad_fn_unprep = backprop.gradients_function(f, [param])
    grad_fn = _prepare(lambda dy, *xs: grad_fn_unprep(*xs, dy=dy), [y_dtype] + [z.dtype for z in xs], [None] + [z.shape for z in xs])
    for row in range(y_size):
        dy_data_flat[row] = 1
        grad = _to_numpy(grad_fn(dy_data, *xs)[0])
        grad = _eval_indexed_slices(grad)
        if isinstance(grad, indexed_slices.IndexedSlicesValue):
            for i, v in zip(grad.indices, grad.values):
                c_begin = i * x_val_size
                c_end = c_begin + x_val_size
                jacobian[row, c_begin:c_end] += v.flat
        elif grad is not None:
            jacobian[row, :] = grad.ravel().view(jacobian.dtype)
        dy_data_flat[row] = 0
    if y_size == 0:
        grad = _to_numpy(grad_fn(dy_data, *xs)[0])
        if grad.shape != x.shape:
            raise ValueError('Empty gradient has wrong shape: expected %s, got %s' % (x.shape, grad.shape))
        if np.any(grad):
            raise ValueError('Empty tensor with nonzero gradients')
    logging.vlog(1, 'Theoretical Jacobian =\n%s', jacobian)
    return jacobian