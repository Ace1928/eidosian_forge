from typing import Any, FrozenSet, Mapping, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq import value
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def _measurement_key_names_from_magic_methods(val: Any) -> Union[FrozenSet[str], NotImplementedType, None]:
    """Uses the measurement key related magic methods to get the key strings for this object."""
    getter = getattr(val, '_measurement_key_names_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return result
    getter = getattr(val, '_measurement_key_name_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return frozenset([result])
    return result