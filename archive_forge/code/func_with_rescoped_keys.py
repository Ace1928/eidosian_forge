from typing import Any, FrozenSet, Mapping, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq import value
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def with_rescoped_keys(val: Any, path: Tuple[str, ...], bindable_keys: Optional[FrozenSet['cirq.MeasurementKey']]=None):
    """Rescopes any measurement and control keys to the provided path, given the existing keys.

    The path usually refers to an identifier or a list of identifiers from a subcircuit that
    used to contain the target. Since a subcircuit can be repeated and reused, these paths help
    differentiate the actual measurement keys.

    This function is generally for internal use in decomposing or iterating subcircuits.

    Args:
        val: The value to rescope.
        path: The prefix to apply to the value's path.
        bindable_keys: The keys that can be bound to at the current scope.
    """
    getter = getattr(val, '_with_rescoped_keys_', None)
    result = NotImplemented if getter is None else getter(path, bindable_keys or frozenset())
    return result if result is not NotImplemented else val