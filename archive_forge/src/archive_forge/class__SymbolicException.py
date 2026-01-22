from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
class _SymbolicException(Exception):
    """Exception class to handle use of symbolic tensors when executing eagerly.

  `keras.Input()` creates symbolic tensors (in a FuncGraph managed by the
  Keras backend) while in eager execution. This exception is used to
  identify this case (raised in `convert_to_tensor` cause generated functions
  for ops to construct graphs instead of executing the kernel).
  """
    pass