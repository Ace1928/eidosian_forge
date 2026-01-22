import collections
from typing import Any, NamedTuple, Optional
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.function.polymorphism import type_dispatch
class FunctionContext(NamedTuple):
    """Contains information regarding tf.function execution context."""
    context: Any = None
    scope_type: Any = None