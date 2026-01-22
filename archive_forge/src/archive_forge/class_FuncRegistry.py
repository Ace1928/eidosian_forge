import functools
import threading
import traceback  # pylint: disable=unused-import
import weakref
import numpy as np
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.lib.core import _pywrap_py_func
from tensorflow.python.ops import autograph_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_script_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
class FuncRegistry:
    """A helper class to keep track of registered py functions.

  FuncRegistry keeps a map from unique tokens (string) to python
  functions, which takes numpy arrays and outputs numpy arrays.
  """

    def __init__(self):
        self._lock = threading.Lock()
        self._unique_id = 0
        self._funcs = weakref.WeakValueDictionary()

    @property
    def _ctx(self):
        context.ensure_initialized()
        return context.context()._handle

    def insert(self, func):
        """Registers `func` and returns a unique token for this entry."""
        token = self._next_unique_token()
        self._funcs[token] = func
        return token

    def remove(self, token):
        """Removes the registered function corresponding to `token`."""
        self._funcs.pop(token, None)

    def get(self, token, default=None):
        """Gets the registered function corresponding to `token`."""
        return self._funcs.get(token, default)

    @staticmethod
    def _convert(value, dtype=None):
        """Converts an arg to numpy, avoiding dangerous string and unicode dtypes.

    Numpy pads with zeros when using string and unicode dtypes if different
    components of a tensor have different lengths.  This is bad: ignoring the
    padding is wrong for text data, and removing the padding is wrong for binary
    data.  To avoid this bug, we redo the conversion using an object dtype.
    Additionally, we convert unicode strings to (byte-)strings for
    compatibility.

    Args:
      value: Value to convert to a numpy array.
      dtype: (Optional.) Desired NumPy type for the returned value.

    Returns:
      A numpy array.
    """
        result = np.asarray(value, dtype=dtype, order='C')
        if result.dtype.char == 'S' and result is not value:
            return np.asarray(value, order='C', dtype=object)
        elif result.dtype.char == 'U' and result is not value:
            value = np.vectorize(lambda x: x.encode('utf8'))(value)
            return np.asarray(value, order='C', dtype=object)
        elif result.dtype.char == 'U':
            return result.astype(np.bytes_)
        else:
            return result

    def __call__(self, token, device, args):
        """Calls the registered function for `token` with args.

    Args:
      token: A key into this `FuncRegistry` identifying which function to call.
      device: Name of the device on which outputs of `token`'s corresponding
        operation should be placed. Used iff the function registered for `token`
        is an EagerPyFunc.
      args: The arguments to pass to the function registered for `token`.

    Returns:
      The output of the function registered for `token`.

    Raises:
      ValueError: if no function is registered for `token`.
    """
        func = self.get(token, None)
        if func is None:
            raise ValueError(f'Could not find callback with key={token} in the registry.')
        if isinstance(func, EagerFunc):
            return func(device, token, args)
        else:
            ret = func(*args)
            if isinstance(ret, bytes):
                ret = [ret]
            if isinstance(ret, (tuple, list)):
                return [self._convert(x) for x in ret]
            else:
                return self._convert(ret)

    def size(self):
        """Returns how many functions are currently registered."""
        return len(self._funcs)

    def _next_unique_token(self):
        """Returns a unique token."""
        with self._lock:
            uid = self._unique_id
            self._unique_id += 1
        return 'pyfunc_%d' % uid