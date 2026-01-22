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
def add_fallback_dispatch_list(target):
    """Decorator that adds a dispatch_list attribute to an op."""
    if hasattr(target, FALLBACK_DISPATCH_ATTR):
        raise AssertionError('%s already has a dispatch list' % target)
    setattr(target, FALLBACK_DISPATCH_ATTR, [])
    return target