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
def type_based_dispatch_signatures_for(cls):
    """Returns dispatch signatures that have been registered for a given class.

  This function is intended for documentation-generation purposes.

  Args:
    cls: The class to search for.  Type signatures are searched recursively, so
      e.g., if `cls=RaggedTensor`, then information will be returned for all
      dispatch targets that have `RaggedTensor` anywhere in their type
      annotations (including nested in `typing.Union` or `typing.List`.)

  Returns:
    A `dict` mapping `api` -> `signatures`, where `api` is a TensorFlow API
    function; and `signatures` is a list of dispatch signatures for `api`
    that include `cls`.  (Each signature is a dict mapping argument names to
    type annotations; see `dispatch_for_api` for more info.)
  """

    def contains_cls(x):
        """Returns true if `x` contains `cls`."""
        if isinstance(x, dict):
            return any((contains_cls(v) for v in x.values()))
        elif x is cls:
            return True
        elif type_annotations.is_generic_list(x) or type_annotations.is_generic_union(x):
            type_args = type_annotations.get_generic_type_args(x)
            return any((contains_cls(arg) for arg in type_args))
        else:
            return False
    result = {}
    for api, api_signatures in _TYPE_BASED_DISPATCH_SIGNATURES.items():
        for _, signatures in api_signatures.items():
            filtered = list(filter(contains_cls, signatures))
            if filtered:
                result.setdefault(api, []).extend(filtered)
    return result