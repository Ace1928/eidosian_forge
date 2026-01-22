from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.util import tf_inspect
def _tf_assert_stmt(expression1, expression2):
    """Overload of assert_stmt that stages a TF Assert.

  This implementation deviates from Python semantics as follows:
    (1) the assertion is verified regardless of the state of __debug__
    (2) on assertion failure, the graph execution will fail with
        tensorflow.errors.ValueError, rather than AssertionError.

  Args:
    expression1: tensorflow.Tensor, must evaluate to a tf.bool scalar
    expression2: Callable[[], Union[tensorflow.Tensor, List[tensorflow.Tensor]]]

  Returns:
    tensorflow.Operation
  """
    expression2_tensors = expression2()
    if not isinstance(expression2_tensors, list):
        expression2_tensors = [expression2_tensors]
    return control_flow_assert.Assert(expression1, expression2_tensors)