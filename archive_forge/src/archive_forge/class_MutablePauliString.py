import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
@value.value_equality(unhashable=True, manual_cls=True, approximate=True)
class MutablePauliString(Generic[TKey]):
    """Mutable version of `cirq.PauliString`, used mainly for efficiently mutating pauli strings.

    `cirq.MutablePauliString` is a mutable version of `cirq.PauliString`, which is often
    useful for mutating pauli strings efficiently instead of always creating a copy. Note
    that, unlike `cirq.PauliString`, `MutablePauliString` is not a `cirq.Operation`.

    It exists mainly to help mutate pauli strings efficiently and then convert back to a
    frozen `cirq.PauliString` representation, which can then be used as operators or
    observables.
    """

    def __init__(self, *contents: 'cirq.PAULI_STRING_LIKE', coefficient: 'cirq.TParamValComplex'=1, pauli_int_dict: Optional[Dict[TKey, int]]=None):
        """Initializes a new `MutablePauliString`.

        Args:
            *contents: A value or values to convert into a pauli string. This
                can be a number, a pauli operation, a dictionary from qubit to
                pauli/identity gates, or collections thereof. If a list of
                values is given, they are each individually converted and then
                multiplied from left to right in order.
            coefficient: Initial scalar coefficient or symbol. Defaults to 1.
            pauli_int_dict: Initial dictionary mapping qubits to integers corresponding
                to pauli operations. Defaults to the empty dictionary. Note that, unlike
                dictionaries passed to contents, this dictionary must not contain values
                corresponding to identity gates; i.e. all integer values must be between
                [1, 3]. Further note that this argument specifies values that are logically
                *before* factors specified in `contents`; `contents` are *right* multiplied
                onto the values in this dictionary.

        Raises:
            ValueError: If the `pauli_int_dict` has integer values `v` not satisfying `1 <= v <= 3`.
        """
        self.coefficient: Union[sympy.Expr, 'cirq.TParamValComplex'] = coefficient if isinstance(coefficient, sympy.Expr) else complex(coefficient)
        if pauli_int_dict is not None:
            for v in pauli_int_dict.values():
                if not 1 <= v <= 3:
                    raise ValueError(f'Value {v} of pauli_int_dict must be between 1 and 3.')
        self.pauli_int_dict: Dict[TKey, int] = {} if pauli_int_dict is None else pauli_int_dict
        if contents:
            self.inplace_left_multiply_by(contents)

    def _value_equality_values_(self):
        return self.frozen()._value_equality_values_()

    def _value_equality_values_cls_(self):
        return self.frozen()._value_equality_values_cls_()

    def _imul_atom_helper(self, key: TKey, pauli_lhs: int, sign: int) -> int:
        pauli_old = self.pauli_int_dict.pop(key, 0)
        pauli_new = pauli_lhs ^ pauli_old
        if pauli_new:
            self.pauli_int_dict[key] = pauli_new
        if not pauli_lhs or not pauli_old or pauli_lhs == pauli_old:
            return 0
        if (pauli_old - pauli_lhs) % 3 == 1:
            return sign
        return -sign

    def keys(self) -> AbstractSet[TKey]:
        """Returns the sequence of qubits on which this pauli string acts."""
        return self.pauli_int_dict.keys()

    def values(self) -> Iterator['cirq.Pauli']:
        """Ordered sequence of `cirq.Pauli` gates acting on `self.keys()`."""
        for v in self.pauli_int_dict.values():
            yield _INT_TO_PAULI[v - 1]

    def __iter__(self) -> Iterator[TKey]:
        return iter(self.pauli_int_dict)

    def __len__(self) -> int:
        return len(self.pauli_int_dict)

    def __bool__(self) -> bool:
        return bool(self.pauli_int_dict)

    def frozen(self) -> 'cirq.PauliString':
        """Returns a `cirq.PauliString` with the same contents.

        For example, this is useful because `cirq.PauliString` is an operation
        whereas `cirq.MutablePauliString` is not.
        """
        return PauliString(coefficient=self.coefficient, qubit_pauli_map={q: _INT_TO_PAULI[p - 1] for q, p in self.pauli_int_dict.items() if p})

    def mutable_copy(self) -> 'cirq.MutablePauliString':
        """Returns a new `cirq.MutablePauliString` with the same contents."""
        return MutablePauliString(coefficient=self.coefficient, pauli_int_dict=dict(self.pauli_int_dict))

    def items(self) -> Iterator[Tuple[TKey, 'cirq.Pauli']]:
        """Returns (cirq.Qid, cirq.Pauli) pairs representing 1-qubit operations of pauli string."""
        for k, v in self.pauli_int_dict.items():
            yield (k, _INT_TO_PAULI[v - 1])

    def __contains__(self, item: Any) -> bool:
        return item in self.pauli_int_dict

    def __getitem__(self, item: Any) -> 'cirq.Pauli':
        return _INT_TO_PAULI[self.pauli_int_dict[item] - 1]

    def __setitem__(self, key: TKey, value: 'cirq.PAULI_GATE_LIKE'):
        value = _pauli_like_to_pauli_int(key, value)
        if value:
            self.pauli_int_dict[key] = _pauli_like_to_pauli_int(key, value)
        else:
            self.pauli_int_dict.pop(key, None)

    def __delitem__(self, key: TKey):
        del self.pauli_int_dict[key]

    @overload
    def get(self, key: TKey, default: None=None) -> Union['cirq.Pauli', None]:
        pass

    @overload
    def get(self, key: TKey, default: TDefault) -> Union['cirq.Pauli', TDefault]:
        pass

    def get(self, key: TKey, default=None) -> Union['cirq.Pauli', TDefault, None]:
        """Returns the `cirq.Pauli` operation acting on qubit `key` or `default` if none exists."""
        result = self.pauli_int_dict.get(key, None)
        return default if result is None else _INT_TO_PAULI[result - 1]

    def inplace_before(self, ops: 'cirq.OP_TREE') -> 'cirq.MutablePauliString':
        """Propagates the pauli string from after to before a Clifford effect.

        If the old value of the MutablePauliString is $P$ and the Clifford
        operation is $C$, then the new value of the MutablePauliString is
        $C^\\dagger P C$.

        Args:
            ops: A stabilizer operation or nested collection of stabilizer
                operations.

        Returns:
            The mutable pauli string that was mutated.
        """
        return self.inplace_after(protocols.inverse(ops))

    def inplace_after(self, ops: 'cirq.OP_TREE') -> 'cirq.MutablePauliString':
        """Propagates the pauli string from before to after a Clifford effect.

        If the old value of the MutablePauliString is $P$ and the Clifford
        operation is $C$, then the new value of the MutablePauliString is
        $C P C^\\dagger$.

        Args:
            ops: A stabilizer operation or nested collection of stabilizer
                operations.

        Returns:
            The mutable pauli string that was mutated.

        Raises:
            NotImplementedError: If any ops decompose into an unsupported
                Clifford gate.
        """
        for clifford in op_tree.flatten_to_ops(ops):
            for op in _decompose_into_cliffords(clifford):
                ps = [self.pauli_int_dict.pop(cast(TKey, q), 0) for q in op.qubits]
                if not any(ps):
                    continue
                gate = op.gate
                if isinstance(gate, clifford_gate.SingleQubitCliffordGate):
                    out = gate.pauli_tuple(_INT_TO_PAULI[ps[0] - 1])
                    if out[1]:
                        self.coefficient *= -1
                    self.pauli_int_dict[cast(TKey, op.qubits[0])] = PAULI_GATE_LIKE_TO_INDEX_MAP[out[0]]
                elif isinstance(gate, pauli_interaction_gate.PauliInteractionGate):
                    q0, q1 = op.qubits
                    p0 = _INT_TO_PAULI_OR_IDENTITY[ps[0]]
                    p1 = _INT_TO_PAULI_OR_IDENTITY[ps[1]]
                    kickback_0_to_1 = not protocols.commutes(p0, gate.pauli0)
                    kickback_1_to_0 = not protocols.commutes(p1, gate.pauli1)
                    kick0 = gate.pauli1 if kickback_0_to_1 else identity.I
                    kick1 = gate.pauli0 if kickback_1_to_0 else identity.I
                    self.__imul__({q0: p0, q1: kick0})
                    self.__imul__({q0: kick1, q1: p1})
                    if gate.invert0:
                        self.inplace_after(gate.pauli1(q1))
                    if gate.invert1:
                        self.inplace_after(gate.pauli0(q0))
                else:
                    raise NotImplementedError(f'Unrecognized decomposed Clifford: {op!r}')
        return self

    def _imul_helper(self, other: 'cirq.PAULI_STRING_LIKE', sign: int):
        """Left-multiplies or right-multiplies by a PAULI_STRING_LIKE.

        Args:
            other: What to multiply by.
            sign: +1 to left-multiply, -1 to right-multiply.

        Returns:
            self on success, NotImplemented given an unknown type of value.
        """
        if isinstance(other, (Mapping, PauliString, MutablePauliString)):
            if isinstance(other, (PauliString, MutablePauliString)):
                self.coefficient *= other.coefficient
            phase_log_i = 0
            for qubit, pauli_gate_like in other.items():
                pauli_int = _pauli_like_to_pauli_int(qubit, pauli_gate_like)
                phase_log_i += self._imul_atom_helper(cast(TKey, qubit), pauli_int, sign)
            self.coefficient *= 1j ** (phase_log_i & 3)
        elif isinstance(other, numbers.Number):
            self.coefficient *= complex(cast(SupportsComplex, other))
        elif isinstance(other, raw_types.Operation) and isinstance(other.gate, identity.IdentityGate):
            pass
        elif isinstance(other, Iterable) and (not isinstance(other, str)) and (not isinstance(other, linear_combinations.PauliSum)):
            if sign == +1:
                other = iter(reversed(list(other)))
            for item in other:
                if self._imul_helper(cast(PAULI_STRING_LIKE, item), sign) is NotImplemented:
                    return NotImplemented
        else:
            return NotImplemented
        return self

    def _imul_helper_checkpoint(self, other: 'cirq.PAULI_STRING_LIKE', sign: int):
        """Like `_imul_helper` but guarantees no-op on error."""
        if not isinstance(other, (numbers.Number, PauliString, MutablePauliString)):
            other = MutablePauliString()._imul_helper(other, sign)
            if other is NotImplemented:
                return NotImplemented
        return self._imul_helper(other, sign)

    def inplace_left_multiply_by(self, other: 'cirq.PAULI_STRING_LIKE') -> 'cirq.MutablePauliString':
        """Left-multiplies a pauli string into this pauli string.

        Args:
            other: A pauli string or `cirq.PAULI_STRING_LIKE` to left-multiply
                into `self`.

        Returns:
            The `self` mutable pauli string that was mutated.

        Raises:
            TypeError: `other` was not a `cirq.PAULI_STRING_LIKE`. `self`
                was not mutated.
        """
        if self._imul_helper_checkpoint(other, -1) is NotImplemented:
            raise TypeError(f'{other!r} is not cirq.PAULI_STRING_LIKE.')
        return self

    def _json_dict_(self) -> Dict[str, Any]:
        return {'pauli_int_dict': list(self.pauli_int_dict.items()), 'coefficient': self.coefficient}

    @classmethod
    def _from_json_dict_(cls, pauli_int_dict, coefficient, **kwargs):
        return cls(pauli_int_dict=dict(pauli_int_dict), coefficient=coefficient)

    def inplace_right_multiply_by(self, other: 'cirq.PAULI_STRING_LIKE') -> 'cirq.MutablePauliString':
        """Right-multiplies a pauli string into this pauli string.

        Args:
            other: A pauli string or `cirq.PAULI_STRING_LIKE` to right-multiply
                into `self`.

        Returns:
            The `self` mutable pauli string that was mutated.

        Raises:
            TypeError: `other` was not a `cirq.PAULI_STRING_LIKE`. `self`
                was not mutated.
        """
        if self._imul_helper_checkpoint(other, +1) is NotImplemented:
            raise TypeError(f'{other!r} is not cirq.PAULI_STRING_LIKE.')
        return self

    def __neg__(self) -> 'cirq.MutablePauliString':
        result = self.mutable_copy()
        result.coefficient *= -1
        return result

    def __pos__(self) -> 'cirq.MutablePauliString':
        return self.mutable_copy()

    def transform_qubits(self, func: Callable[[TKey], TKeyNew], *, inplace: bool=False) -> 'cirq.MutablePauliString[TKeyNew]':
        """Returns a `MutablePauliString` with transformed qubits.

        Args:
            func: The qubit transformation to apply.
            inplace: If false (the default), creates a new mutable pauli string
                to store the result. If true, overwrites this mutable pauli
                string's contents. Defaults to false for consistency with
                `cirq.PauliString.transform_qubits` in situations where the
                pauli string being used may or may not be mutable.

        Returns:
            A transformed MutablePauliString.
            If inplace=True, returns `self`.
            If inplace=False, returns a new instance.
        """
        new_dict = {func(q): p for q, p in self.pauli_int_dict.items()}
        if not inplace:
            return MutablePauliString(coefficient=self.coefficient, pauli_int_dict=new_dict)
        result = cast('cirq.MutablePauliString[TKeyNew]', self)
        result.pauli_int_dict = new_dict
        return result

    def __imul__(self, other: 'cirq.PAULI_STRING_LIKE') -> 'cirq.MutablePauliString':
        """Left-multiplies a pauli string into this pauli string.

        Args:
            other: A pauli string or `cirq.PAULI_STRING_LIKE` to left-multiply
                into `self`.

        Returns:
            The `self` mutable pauli string that was successfully mutated.

            If `other` is not a `cirq.PAULI_STRING_LIKE`, `self` is not mutated
            and `NotImplemented` is returned.

        """
        return self._imul_helper_checkpoint(other, +1)

    def __mul__(self, other: 'cirq.PAULI_STRING_LIKE') -> 'cirq.PauliString':
        """Multiplies two pauli-string-likes together.

        The result is not mutable.
        """
        return self.frozen() * other

    def __rmul__(self, other: 'cirq.PAULI_STRING_LIKE') -> 'cirq.PauliString':
        """Multiplies two pauli-string-likes together.

        The result is not mutable.
        """
        return other * self.frozen()

    def __str__(self) -> str:
        return f'mutable {self.frozen()}'

    def __repr__(self) -> str:
        return f'{self.frozen()!r}.mutable_copy()'