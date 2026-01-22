from typing import Any, FrozenSet, Mapping, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq import value
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def _measurement_key_objs_from_magic_methods(val: Any) -> Union[FrozenSet['cirq.MeasurementKey'], NotImplementedType, None]:
    """Uses the measurement key related magic methods to get the `MeasurementKey`s for this
    object."""
    getter = getattr(val, '_measurement_key_objs_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return result
    getter = getattr(val, '_measurement_key_obj_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return frozenset([result])
    return result