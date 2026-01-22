import collections
from typing import Optional, Iterable
from tensorflow.core.function.polymorphism import function_type
def add_target(self, target: function_type.FunctionType) -> None:
    """Adds a new target type."""
    self._dispatch_table[target] = None
    for request in self._dispatch_cache:
        if target.is_supertype_of(self._dispatch_cache[request]):
            self._dispatch_cache[request] = target