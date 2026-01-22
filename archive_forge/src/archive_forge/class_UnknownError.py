import traceback
import warnings
from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python import _pywrap_py_exception_registry
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export('errors.UnknownError')
class UnknownError(OpError):
    """Unknown error.

  An example of where this error may be returned is if a Status value
  received from another address space belongs to an error-space that
  is not known to this address space. Also, errors raised by APIs that
  do not return enough error information may be converted to this
  error.
  """

    def __init__(self, node_def, op, message, *args):
        """Creates an `UnknownError`."""
        super(UnknownError, self).__init__(node_def, op, message, UNKNOWN, *args)