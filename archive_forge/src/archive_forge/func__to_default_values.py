import functools
import inspect
from typing import Any, Dict, Tuple
import six
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import nest
def _to_default_values(fullargspec):
    """Returns default values from the function's inspected fullargspec."""
    if fullargspec.defaults is not None:
        defaults = {name: value for name, value in zip(fullargspec.args[-len(fullargspec.defaults):], fullargspec.defaults)}
    else:
        defaults = {}
    if fullargspec.kwonlydefaults is not None:
        defaults.update(fullargspec.kwonlydefaults)
    defaults = {function_type_lib.sanitize_arg_name(name): value for name, value in defaults.items()}
    return defaults