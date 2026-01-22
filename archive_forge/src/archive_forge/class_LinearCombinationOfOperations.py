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
class LinearCombinationOfOperations(value.LinearDict[raw_types.Operation]):
    """Represents operator defined by linear combination of gate operations.

    If G1, ..., Gn are gate operations, {q1_1, ..., q1_k1}, {q2_1, ..., q2_k2},
    ..., {qn_1, ..., qn_kn} are (not necessarily disjoint) sets of qubits and
    b1, b2, ..., bn are complex numbers, then

        LinearCombinationOfOperations({
            G1(q1_1, ..., q1_k1): b1,
            G2(q2_1, ..., q2_k2): b2,
            ...,
            Gn(qn_1, ..., qn_kn): bn})

    represents the linear operator

        A = b1 G1(q1_1, ..., q1_k1) +
          + b2 G2(q2_1, ..., q2_k2) +
          + ... +
          + bn Gn(qn_1, ..., qn_kn)

    where in each term qubits not explicitly listed are assumed to be acted on
    by the identity operator. Note that A may not be unitary or even normal.
    """

    def __init__(self, terms: Mapping[raw_types.Operation, value.Scalar]) -> None:
        """Initializes linear combination from a collection of terms.

        Args:
            terms: Mapping of gate operations to coefficients in the linear
                combination being initialized.
        """
        super().__init__(terms, validator=self._is_compatible)

    def _is_compatible(self, operation: 'cirq.Operation') -> bool:
        return isinstance(operation, raw_types.Operation)

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        """Returns qubits acted on self."""
        if not self:
            return ()
        qubit_sets = [set(op.qubits) for op in self.keys()]
        all_qubits = set.union(*qubit_sets)
        return tuple(sorted(all_qubits))

    def __pow__(self, exponent: int) -> 'LinearCombinationOfOperations':
        if not isinstance(exponent, int):
            return NotImplemented
        if exponent < 0:
            return NotImplemented
        if len(self.qubits) != 1:
            return NotImplemented
        qubit = self.qubits[0]
        i = identity.I(qubit)
        x = pauli_gates.X(qubit)
        y = pauli_gates.Y(qubit)
        z = pauli_gates.Z(qubit)
        pauli_basis = {i, x, y, z}
        if not set(self.keys()).issubset(pauli_basis):
            return NotImplemented
        ai, ax, ay, az = (self[i], self[x], self[y], self[z])
        bi, bx, by, bz = operator_spaces.pow_pauli_combination(ai, ax, ay, az, exponent)
        return LinearCombinationOfOperations({i: bi, x: bx, y: by, z: bz})

    def _is_parameterized_(self) -> bool:
        return any((protocols.is_parameterized(op) for op in self.keys()))

    def _parameter_names_(self) -> AbstractSet[str]:
        return {name for op in self.keys() for name in protocols.parameter_names(op)}

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'LinearCombinationOfOperations':
        return self.__class__({protocols.resolve_parameters(op, resolver, recursive): coeff for op, coeff in self.items()})

    def matrix(self) -> np.ndarray:
        """Reconstructs matrix of self using unitaries of underlying operations.

        Raises:
            TypeError: if any of the gates in self does not provide a unitary.
        """
        if self._is_parameterized_():
            return NotImplemented
        num_qubits = len(self.qubits)
        num_dim = 2 ** num_qubits
        qubit_to_axis = {q: i for i, q in enumerate(self.qubits)}
        result = np.zeros((2,) * (2 * num_qubits), dtype=np.complex128)
        for op, coefficient in self.items():
            identity = np.eye(num_dim, dtype=np.complex128).reshape(result.shape)
            workspace = np.empty_like(identity)
            axes = tuple((qubit_to_axis[q] for q in op.qubits))
            u = protocols.apply_unitary(op, protocols.ApplyUnitaryArgs(identity, workspace, axes))
            result += coefficient * u
        return result.reshape((num_dim, num_dim))

    def _has_unitary_(self) -> bool:
        m = self.matrix()
        return m is not NotImplemented and linalg.is_unitary(m)

    def _unitary_(self) -> np.ndarray:
        m = self.matrix()
        if m is NotImplemented or linalg.is_unitary(m):
            return m
        raise ValueError(f'{self} is not unitary')

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        """Computes Pauli expansion of self from Pauli expansions of terms."""

        def extend_term(pauli_names: str, qubits: Tuple['cirq.Qid', ...], all_qubits: Tuple['cirq.Qid', ...]) -> str:
            """Extends Pauli product on qubits to product on all_qubits."""
            assert len(pauli_names) == len(qubits)
            qubit_to_pauli_name = dict(zip(qubits, pauli_names))
            return ''.join((qubit_to_pauli_name.get(q, 'I') for q in all_qubits))

        def extend(expansion: value.LinearDict[str], qubits: Tuple['cirq.Qid', ...], all_qubits: Tuple['cirq.Qid', ...]) -> value.LinearDict[str]:
            """Extends Pauli expansion on qubits to expansion on all_qubits."""
            return value.LinearDict({extend_term(p, qubits, all_qubits): c for p, c in expansion.items()})
        result: value.LinearDict[str] = value.LinearDict({})
        for op, coefficient in self.items():
            expansion = protocols.pauli_expansion(op)
            extended_expansion = extend(expansion, op.qubits, self.qubits)
            result += extended_expansion * coefficient
        return result