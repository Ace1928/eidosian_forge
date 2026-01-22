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
def _remove_annotation(sig):
    """Removes annotation from a python Signature."""
    parameters = [p.replace(annotation=p.empty) for p in sig.parameters.values()]
    return sig.replace(parameters=parameters, return_annotation=sig.empty)