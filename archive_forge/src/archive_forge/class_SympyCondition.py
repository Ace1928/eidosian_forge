import abc
import dataclasses
from typing import Mapping, Tuple, TYPE_CHECKING, FrozenSet
import sympy
from cirq._compat import proper_repr
from cirq.protocols import json_serialization, measurement_key_protocol as mkp
from cirq.value import measurement_key
@dataclasses.dataclass(frozen=True)
class SympyCondition(Condition):
    """A classical control condition based on a sympy expression.

    This condition resolves to True iff the sympy expression resolves to a
    truthy value (i.e. `bool(x) == True`) when the measurement keys are
    substituted in as the free variables.
    """
    expr: sympy.Basic

    @property
    def keys(self):
        return tuple((measurement_key.MeasurementKey.parse_serialized(symbol.name) for symbol in self.expr.free_symbols))

    def replace_key(self, current: 'cirq.MeasurementKey', replacement: 'cirq.MeasurementKey'):
        return SympyCondition(self.expr.subs({str(current): sympy.Symbol(str(replacement))}))

    def __str__(self):
        return str(self.expr)

    def __repr__(self):
        return f'cirq.SympyCondition({proper_repr(self.expr)})'

    def resolve(self, classical_data: 'cirq.ClassicalDataStoreReader') -> bool:
        missing = [str(k) for k in self.keys if k not in classical_data.keys()]
        if missing:
            raise ValueError(f'Measurement keys {missing} missing when testing classical control')
        replacements = {str(k): classical_data.get_int(k) for k in self.keys}
        return bool(self.expr.subs(replacements))

    def _json_dict_(self):
        return json_serialization.dataclass_json_dict(self)

    @classmethod
    def _from_json_dict_(cls, expr, **kwargs):
        return cls(expr=expr)

    @property
    def qasm(self):
        if isinstance(self.expr, sympy.Equality):
            if isinstance(self.expr.lhs, sympy.Symbol) and isinstance(self.expr.rhs, sympy.Integer):
                return f'm_{self.expr.lhs}=={self.expr.rhs}'
        raise ValueError('QASM is defined only for SympyConditions of type key == constant.')