from collections import defaultdict
from typing import (
import numbers
import numpy as np
from sympy.logic.boolalg import And, Not, Or, Xor
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from scipy.sparse import csr_matrix
from cirq import linalg, protocols, qis, value
from cirq._doc import document
from cirq.linalg import operator_spaces
from cirq.ops import identity, raw_types, pauli_gates, pauli_string
from cirq.ops.pauli_string import PauliString, _validate_qubit_mapping
from cirq.ops.projector import ProjectorString
from cirq.value.linear_dict import _format_terms
class LinearCombinationOfGates(value.LinearDict[raw_types.Gate]):
    """Represents linear operator defined by a linear combination of gates.

    Suppose G1, G2, ..., Gn are gates and b1, b2, ..., bn are complex
    numbers. Then

        LinearCombinationOfGates({G1: b1, G2: b2, ..., Gn: bn})

    represents the linear operator

        A = b1 G1 + b2 G2 + ... + bn Gn

    Note that A may not be unitary or even normal.

    Rather than creating LinearCombinationOfGates instance explicitly, one may
    use overloaded arithmetic operators. For example,

        cirq.LinearCombinationOfGates({cirq.X: 2, cirq.Z: -2})

    is equivalent to

        2 * cirq.X - 2 * cirq.Z
    """

    def __init__(self, terms: Mapping[raw_types.Gate, value.Scalar]) -> None:
        """Initializes linear combination from a collection of terms.

        Args:
            terms: Mapping of gates to coefficients in the linear combination
                being initialized.
        """
        super().__init__(terms, validator=self._is_compatible)

    def num_qubits(self) -> Optional[int]:
        """Returns number of qubits in the domain if known, None if unknown."""
        if not self:
            return None
        any_gate = next(iter(self))
        return any_gate.num_qubits()

    def _is_compatible(self, gate: 'cirq.Gate') -> bool:
        return self.num_qubits() is None or self.num_qubits() == gate.num_qubits()

    def __add__(self, other: Union[raw_types.Gate, 'LinearCombinationOfGates']) -> 'LinearCombinationOfGates':
        if not isinstance(other, LinearCombinationOfGates):
            other = other.wrap_in_linear_combination()
        return super().__add__(other)

    def __iadd__(self, other: Union[raw_types.Gate, 'LinearCombinationOfGates']) -> 'LinearCombinationOfGates':
        if not isinstance(other, LinearCombinationOfGates):
            other = other.wrap_in_linear_combination()
        return super().__iadd__(other)

    def __sub__(self, other: Union[raw_types.Gate, 'LinearCombinationOfGates']) -> 'LinearCombinationOfGates':
        if not isinstance(other, LinearCombinationOfGates):
            other = other.wrap_in_linear_combination()
        return super().__sub__(other)

    def __isub__(self, other: Union[raw_types.Gate, 'LinearCombinationOfGates']) -> 'LinearCombinationOfGates':
        if not isinstance(other, LinearCombinationOfGates):
            other = other.wrap_in_linear_combination()
        return super().__isub__(other)

    def __pow__(self, exponent: int) -> 'LinearCombinationOfGates':
        if not isinstance(exponent, int):
            return NotImplemented
        if exponent < 0:
            return NotImplemented
        if self.num_qubits() != 1:
            return NotImplemented
        pauli_basis = {identity.I, pauli_gates.X, pauli_gates.Y, pauli_gates.Z}
        if not set(self.keys()).issubset(pauli_basis):
            return NotImplemented
        ai = self[identity.I]
        ax = self[pauli_gates.X]
        ay = self[pauli_gates.Y]
        az = self[pauli_gates.Z]
        bi, bx, by, bz = operator_spaces.pow_pauli_combination(ai, ax, ay, az, exponent)
        return LinearCombinationOfGates({identity.I: bi, pauli_gates.X: bx, pauli_gates.Y: by, pauli_gates.Z: bz})

    def _is_parameterized_(self) -> bool:
        return any((protocols.is_parameterized(gate) for gate in self.keys()))

    def _parameter_names_(self) -> AbstractSet[str]:
        return {name for gate in self.keys() for name in protocols.parameter_names(gate)}

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'LinearCombinationOfGates':
        return self.__class__({protocols.resolve_parameters(gate, resolver, recursive): coeff for gate, coeff in self.items()})

    def matrix(self) -> np.ndarray:
        """Reconstructs matrix of self using unitaries of underlying gates.

        Raises:
            ValueError: If the number of qubits has not been specified.
        """
        if self._is_parameterized_():
            return NotImplemented
        num_qubits = self.num_qubits()
        if num_qubits is None:
            raise ValueError('Unknown number of qubits')
        num_dim = 2 ** num_qubits
        result = np.zeros((num_dim, num_dim), dtype=np.complex128)
        for gate, coefficient in self.items():
            result += protocols.unitary(gate) * coefficient
        return result

    def _has_unitary_(self) -> bool:
        m = self.matrix()
        return m is not NotImplemented and linalg.is_unitary(m)

    def _unitary_(self) -> np.ndarray:
        m = self.matrix()
        if m is NotImplemented or linalg.is_unitary(m):
            return m
        raise ValueError(f'{self} is not unitary')

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        result: value.LinearDict[str] = value.LinearDict({})
        for gate, coefficient in self.items():
            result += protocols.pauli_expansion(gate) * coefficient
        return result