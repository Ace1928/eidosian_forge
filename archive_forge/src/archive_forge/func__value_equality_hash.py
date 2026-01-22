from typing import Any, Callable, Optional, overload, Union
from typing_extensions import Protocol
from cirq import protocols, _compat
def _value_equality_hash(self: _SupportsValueEquality) -> int:
    return hash((self._value_equality_values_cls_(), self._value_equality_values_()))