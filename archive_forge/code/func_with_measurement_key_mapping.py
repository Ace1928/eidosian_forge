from typing import Any, FrozenSet, Mapping, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq import value
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def with_measurement_key_mapping(val: Any, key_map: Mapping[str, str]):
    """Remaps the target's measurement keys according to the provided key_map.

    This method can be used to reassign measurement keys at runtime, or to
    assign measurement keys from a higher-level object (such as a Circuit).
    """
    getter = getattr(val, '_with_measurement_key_mapping_', None)
    return NotImplemented if getter is None else getter(key_map)