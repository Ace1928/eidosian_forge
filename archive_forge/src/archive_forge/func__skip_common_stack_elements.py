import functools
import traceback
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def _skip_common_stack_elements(stacktrace, base_case):
    """Skips items that the target stacktrace shares with the base stacktrace."""
    for i, (trace, base) in enumerate(zip(stacktrace, base_case)):
        if trace != base:
            return stacktrace[i:]
    return stacktrace[-1:]