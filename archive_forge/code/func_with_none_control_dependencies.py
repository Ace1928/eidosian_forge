import collections.abc
from typing import Any, Hashable, Optional, Dict
import weakref
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
@property
def with_none_control_dependencies(self) -> bool:
    return self._with_none_control_dependencies