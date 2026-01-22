from typing import Any, Callable, Optional, overload, Union
from typing_extensions import Protocol
from cirq import protocols, _compat
def _value_equality_eq(self: _SupportsValueEquality, other: _SupportsValueEquality) -> bool:
    cls_self = self._value_equality_values_cls_()
    get_cls_other = getattr(other, '_value_equality_values_cls_', None)
    if get_cls_other is None:
        return NotImplemented
    cls_other = other._value_equality_values_cls_()
    if cls_self != cls_other:
        return False
    return self._value_equality_values_() == other._value_equality_values_()