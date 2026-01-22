import collections
import itertools
import typing  # pylint: disable=unused-import (used in doctests)
from tensorflow.python.framework import _pywrap_python_api_dispatcher as _api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export as tf_export_lib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import type_annotations
from tensorflow.python.util.tf_export import tf_export
def _check_signature(api_signature, func):
    """Checks that a dispatch target's signature is compatible with an API.

  Args:
    api_signature: The signature of the TensorFlow API.
    func: The dispatch target.

  Raises:
    ValueError: if the signatures are incompatible.  Two signatures are
      considered compatible if they have the same number of parameters, and all
      corresponding parameters have the same `name` and `kind`.  (Parameters
      are not required to have the same default value or the same annotation.)
  """
    func_argspec = tf_inspect.getargspec(func)
    if func_argspec.varargs is not None and func_argspec.keywords is not None and (not func_argspec.args):
        return
    func_signature = tf_inspect.signature(func)
    ok = len(api_signature.parameters) == len(func_signature.parameters)
    if ok:
        for param_1, param_2 in zip(api_signature.parameters.values(), func_signature.parameters.values()):
            if param_1.name != param_2.name or param_1.kind != param_2.kind:
                ok = False
    if not ok:
        raise ValueError(f"Dispatch function's signature {func_signature} does not match API's signature {api_signature}.")