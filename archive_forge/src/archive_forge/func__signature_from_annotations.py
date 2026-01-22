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
def _signature_from_annotations(func):
    """Builds a dict mapping from parameter names to type annotations."""
    func_signature = tf_inspect.signature(func)
    signature = dict([(name, param.annotation) for name, param in func_signature.parameters.items() if param.annotation != tf_inspect.Parameter.empty])
    if not signature:
        raise ValueError('The dispatch_for_api decorator must be called with at least one signature, or applied to a function that has type annotations on its parameters.')
    return signature