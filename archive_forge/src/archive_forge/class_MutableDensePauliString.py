import abc
import numbers
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, linalg, value
from cirq._compat import proper_repr
from cirq.ops import raw_types, identity, pauli_gates, global_phase_op, pauli_string
from cirq.type_workarounds import NotImplementedType
@value.value_equality(unhashable=True, approximate=True)
class MutableDensePauliString(BaseDensePauliString):
    """A mutable string of Paulis, like `XIXY`, with a coefficient.

    `cirq.MutableDensePauliString` is a mutable version of `cirq.DensePauliString`.
    It exists mainly to help mutate dense pauli strings efficiently, instead of always creating
    a copy, and then converting back to a frozen `cirq.DensePauliString` representation.

    For example:

    >>> mutable_dps = cirq.MutableDensePauliString('XXZZ')
    >>> mutable_dps[:2] = 'YY' # `cirq.MutableDensePauliString` supports item assignment.
    >>> print(mutable_dps)
    +YYZZ (mutable)

    See docstrings of `cirq.DensePauliString` for more details on dense pauli strings.
    """

    @overload
    def __setitem__(self, key: int, value: 'cirq.PAULI_GATE_LIKE') -> Self:
        pass

    @overload
    def __setitem__(self, key: slice, value: Union[Iterable['cirq.PAULI_GATE_LIKE'], np.ndarray, BaseDensePauliString]) -> Self:
        pass

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.pauli_mask[key] = _pauli_index(value)
            return self
        if isinstance(key, slice):
            if isinstance(value, BaseDensePauliString):
                if value.coefficient != 1:
                    raise ValueError("Can't slice-assign from a PauliProduct whose coefficient is not 1.\n\nWorkaround: If you just want to ignore the coefficient, do `= value[:]` instead of `= value`.")
                self.pauli_mask[key] = value.pauli_mask
            else:
                self.pauli_mask[key] = _as_pauli_mask(value)
            return self
        raise TypeError(f'indices must be integers or slices, not {type(key)}')

    def __itruediv__(self, other):
        if isinstance(other, (sympy.Basic, numbers.Number)):
            return self.__imul__(1 / other)
        return NotImplemented

    def __imul__(self, other):
        if isinstance(other, BaseDensePauliString):
            if len(other) > len(self):
                raise ValueError(f'The receiving dense pauli string is smaller than the dense pauli string being multiplied into it.\nself={repr(self)}\nother={repr(other)}')
            self_mask = self.pauli_mask[:len(other.pauli_mask)]
            self._coefficient *= _vectorized_pauli_mul_phase(self_mask, other.pauli_mask)
            self._coefficient *= other.coefficient
            self_mask ^= other.pauli_mask
            return self
        if isinstance(other, (sympy.Basic, numbers.Number)):
            new_coef = protocols.mul(self.coefficient, other, default=None)
            if new_coef is None:
                return NotImplemented
            self._coefficient = new_coef if isinstance(new_coef, sympy.Basic) else complex(new_coef)
            return self
        split = _attempt_value_to_pauli_index(other)
        if split is not None:
            p, i = split
            self._coefficient *= _vectorized_pauli_mul_phase(self.pauli_mask[i], p)
            self.pauli_mask[i] ^= p
            return self
        return NotImplemented

    def copy(self, coefficient: Optional[Union[sympy.Expr, int, float, complex]]=None, pauli_mask: Union[None, str, Iterable[int], np.ndarray]=None) -> 'MutableDensePauliString':
        return MutableDensePauliString(coefficient=self.coefficient if coefficient is None else coefficient, pauli_mask=np.copy(self.pauli_mask) if pauli_mask is None else pauli_mask)

    def __str__(self) -> str:
        return super().__str__() + ' (mutable)'

    def _value_equality_values_(self):
        return (self.coefficient, tuple((PAULI_CHARS[p] for p in self.pauli_mask)))

    @classmethod
    def inline_gaussian_elimination(cls, rows: 'List[MutableDensePauliString]') -> None:
        if not rows:
            return
        height = len(rows)
        width = len(rows[0])
        next_row = 0
        for col in range(width):
            for held in [DensePauliString.Z_VAL, DensePauliString.X_VAL]:
                for k in range(next_row, height):
                    if (rows[k].pauli_mask[col] or held) != held:
                        pivot_row = k
                        break
                else:
                    continue
                for k in range(height):
                    if k != pivot_row:
                        if (rows[k].pauli_mask[col] or held) != held:
                            rows[k].__imul__(rows[pivot_row])
                if pivot_row != next_row:
                    rows[next_row], rows[pivot_row] = (rows[pivot_row], rows[next_row])
                next_row += 1