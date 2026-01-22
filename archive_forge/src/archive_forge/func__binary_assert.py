import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _binary_assert(sym, opname, op_func, static_func, x, y, data, summarize, message, name):
    """Generic binary elementwise assertion.

  Implements the behavior described in _binary_assert_doc() above.
  Args:
    sym: Mathematical symbol for the test to apply to pairs of tensor elements,
      i.e. "=="
    opname: Name of the assert op in the public API, i.e. "assert_equal"
    op_func: Function that, if passed the two Tensor inputs to the assertion (x
      and y), will return the test to be passed to reduce_all() i.e.
    static_func: Function that, if passed numpy ndarray versions of the two
      inputs to the assertion, will return a Boolean ndarray with containing
      True in all positions where the assertion PASSES.
      i.e. np.equal for assert_equal()
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to the value of
      `opname`.

  Returns:
    See docstring template in _binary_assert_doc().
  """
    with ops.name_scope(name, opname, [x, y, data]):
        x = ops.convert_to_tensor(x, name='x')
        y = ops.convert_to_tensor(y, name='y')
        if context.executing_eagerly():
            test_op = op_func(x, y)
            condition = math_ops.reduce_all(test_op)
            if condition:
                return
            if summarize is None:
                summarize = 3
            elif summarize < 0:
                summarize = 1000000000.0
            if data is None:
                data = _make_assert_msg_data(sym, x, y, summarize, test_op)
            if message is not None:
                data = [message] + list(data)
            raise errors.InvalidArgumentError(node_def=None, op=None, message='\n'.join((_pretty_print(d, summarize) for d in data)))
        else:
            if data is None:
                data = ['Condition x %s y did not hold element-wise:' % sym, 'x (%s) = ' % x.name, x, 'y (%s) = ' % y.name, y]
            if message is not None:
                data = [message] + list(data)
            condition = math_ops.reduce_all(op_func(x, y))
            x_static = tensor_util.constant_value(x)
            y_static = tensor_util.constant_value(y)
            if x_static is not None and y_static is not None:
                condition_static = np.all(static_func(x_static, y_static))
                _assert_static(condition_static, data)
            return control_flow_assert.Assert(condition, data, summarize=summarize)