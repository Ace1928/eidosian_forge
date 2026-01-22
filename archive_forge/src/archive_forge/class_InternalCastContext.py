import collections.abc
from typing import Any, Hashable, Optional, Dict
import weakref
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
class InternalCastContext(trace.CastContext):
    """Default casting behaviors."""

    def __init__(self, allow_specs=False):
        self._allow_specs = allow_specs

    @property
    def allow_specs(self) -> bool:
        """Allow TypeSpecs to be casted (instead of the actual CompositeTensors)."""
        return self._allow_specs