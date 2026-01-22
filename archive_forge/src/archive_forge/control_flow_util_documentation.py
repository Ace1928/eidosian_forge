from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import variables
Return the bool value for `pred`, or None if `pred` had a dynamic value.

  Args:
    pred: A scalar, either a Python bool or a TensorFlow boolean variable
      or tensor, or the Python integer 1 or 0.

  Returns:
    True or False if `pred` has a constant boolean value, None otherwise.

  Raises:
    TypeError: If `pred` is not a Variable, Tensor or bool, or Python
      integer 1 or 0.
  