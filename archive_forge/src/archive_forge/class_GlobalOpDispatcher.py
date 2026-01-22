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
@tf_export('__internal__.dispatch.GlobalOpDispatcher', v1=[])
class GlobalOpDispatcher(object):
    """Abstract base class for TensorFlow global operator dispatchers."""
    NOT_SUPPORTED = OpDispatcher.NOT_SUPPORTED

    def handle(self, op, args, kwargs):
        """Handle the specified operation with the specified arguments."""

    def register(self):
        """Register this dispatcher as a handler for all ops."""
        _GLOBAL_DISPATCHERS.append(self)