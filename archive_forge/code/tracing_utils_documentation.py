from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.checkpoint import tensor_callable
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
Traces `Trackable` serialize- and restore-from-tensors functions.

  Args:
    obj: A `Trackable` object.

  Returns:
    A concrete Function.
  