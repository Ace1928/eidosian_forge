import collections
from typing import Any, NamedTuple, Optional
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.function.polymorphism import type_dispatch
class FunctionCache:
    """A container for managing functions."""
    __slots__ = ['_primary', '_dispatch_dict', '_garbage_collectors']

    def __init__(self):
        self._primary = collections.OrderedDict()
        self._dispatch_dict = {}

    def lookup(self, function_type: function_type_lib.FunctionType, context: Optional[FunctionContext]=None) -> Optional[Any]:
        """Looks up a function based on the context and type."""
        context = context or FunctionContext()
        if context in self._dispatch_dict:
            dispatch_type = self._dispatch_dict[context].dispatch(function_type)
            if dispatch_type:
                return self._primary[context, dispatch_type]
        return None

    def delete(self, function_type: function_type_lib.FunctionType, context: Optional[FunctionContext]=None) -> bool:
        """Deletes a function given the context and type."""
        context = context or FunctionContext()
        if (context, function_type) not in self._primary:
            return False
        del self._primary[context, function_type]
        self._dispatch_dict[context].delete(function_type)
        return True

    def add(self, fn: Any, context: Optional[FunctionContext]=None) -> None:
        """Adds a new function using its function_type.

    Args:
      fn: The function to be added to the cache.
      context: A FunctionContext representing the current context.
    """
        context = context or FunctionContext()
        self._primary[context, fn.function_type] = fn
        if context not in self._dispatch_dict:
            self._dispatch_dict[context] = type_dispatch.TypeDispatchTable()
        self._dispatch_dict[context].add_target(fn.function_type)

    def generalize(self, context: FunctionContext, function_type: function_type_lib.FunctionType) -> function_type_lib.FunctionType:
        """Try to generalize a FunctionType within a FunctionContext."""
        if context in self._dispatch_dict:
            return self._dispatch_dict[context].try_generalizing_function_type(function_type)
        else:
            return function_type

    def clear(self):
        """Removes all functions from the cache."""
        self._primary.clear()
        self._dispatch_dict.clear()

    def values(self):
        """Returns a list of all functions held by this cache."""
        return list(self._primary.values())

    def __len__(self):
        return len(self._primary)