from typing import Any, FrozenSet, Mapping, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq import value
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def with_key_path(val: Any, path: Tuple[str, ...]):
    """Adds the path to the target's measurement keys.

    The path usually refers to an identifier or a list of identifiers from a subcircuit that
    used to contain the target. Since a subcircuit can be repeated and reused, these paths help
    differentiate the actual measurement keys.
    """
    getter = getattr(val, '_with_key_path_', None)
    return NotImplemented if getter is None else getter(path)