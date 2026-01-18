from tensorflow.core.framework import kernel_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
Returns a KernelList proto of registered kernels for a given op.

  Args:
    name: A string representing the name of the op whose kernels to retrieve.
  