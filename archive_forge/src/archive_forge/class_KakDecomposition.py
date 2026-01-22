import cmath
import math
from typing import (
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq._import import LazyLoader
from cirq.linalg import combinators, diagonalize, predicates, transformations
@value.value_equality
class KakDecomposition:
    """A convenient description of an arbitrary two-qubit operation.

    Any two qubit operation U can be decomposed into the form

        U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)

    This class stores g, (b0, b1), (x, y, z), and (a0, a1).

    Attributes:
        global_phase: g from the above equation.
        single_qubit_operations_before: b0, b1 from the above equation.
        interaction_coefficients: x, y, z from the above equation.
        single_qubit_operations_after: a0, a1 from the above equation.

    References:
        'An Introduction to Cartan's KAK Decomposition for QC Programmers'
        https://arxiv.org/abs/quant-ph/0507171
    """

    def __init__(self, *, global_phase: complex=complex(1), single_qubit_operations_before: Optional[Tuple[np.ndarray, np.ndarray]]=None, interaction_coefficients: Tuple[float, float, float], single_qubit_operations_after: Optional[Tuple[np.ndarray, np.ndarray]]=None):
        """Initializes a decomposition for a two-qubit operation U.

        U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)

        Args:
            global_phase: g from the above equation.
            single_qubit_operations_before: b0, b1 from the above equation.
            interaction_coefficients: x, y, z from the above equation.
            single_qubit_operations_after: a0, a1 from the above equation.
        """
        self.global_phase: complex = global_phase
        self.single_qubit_operations_before: Tuple[np.ndarray, np.ndarray] = single_qubit_operations_before or (np.eye(2, dtype=np.complex64), np.eye(2, dtype=np.complex64))
        self.interaction_coefficients = interaction_coefficients
        self.single_qubit_operations_after: Tuple[np.ndarray, np.ndarray] = single_qubit_operations_after or (np.eye(2, dtype=np.complex64), np.eye(2, dtype=np.complex64))

    def _value_equality_values_(self) -> Any:

        def flatten(x):
            return tuple((tuple(e.flat) for e in x))
        return (self.global_phase, tuple(self.interaction_coefficients), flatten(self.single_qubit_operations_before), flatten(self.single_qubit_operations_after))

    def __str__(self) -> str:
        xx = self.interaction_coefficients[0] * 4 / np.pi
        yy = self.interaction_coefficients[1] * 4 / np.pi
        zz = self.interaction_coefficients[2] * 4 / np.pi
        before0 = axis_angle(self.single_qubit_operations_before[0])
        before1 = axis_angle(self.single_qubit_operations_before[1])
        after0 = axis_angle(self.single_qubit_operations_after[0])
        after1 = axis_angle(self.single_qubit_operations_after[1])
        return f'KAK {{\n    xyz*(4/π): {xx:.3g}, {yy:.3g}, {zz:.3g}\n    before: ({before0}) ⊗ ({before1})\n    after: ({after0}) ⊗ ({after1})\n}}'

    def __repr__(self) -> str:
        before0 = proper_repr(self.single_qubit_operations_before[0])
        before1 = proper_repr(self.single_qubit_operations_before[1])
        after0 = proper_repr(self.single_qubit_operations_after[0])
        after1 = proper_repr(self.single_qubit_operations_after[1])
        return f'cirq.KakDecomposition(\n    interaction_coefficients={self.interaction_coefficients!r},\n    single_qubit_operations_before=(\n        {before0},\n        {before1},\n    ),\n    single_qubit_operations_after=(\n        {after0},\n        {after1},\n    ),\n    global_phase={self.global_phase!r})'

    def _unitary_(self) -> np.ndarray:
        """Returns the decomposition's two-qubit unitary matrix.

        U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)
        """
        before = np.kron(*self.single_qubit_operations_before)
        after = np.kron(*self.single_qubit_operations_after)

        def interaction_matrix(m: np.ndarray, c: float) -> np.ndarray:
            return map_eigenvalues(np.kron(m, m), lambda v: np.exp(1j * v * c))
        x, y, z = self.interaction_coefficients
        x_mat = np.array([[0, 1], [1, 0]])
        y_mat = np.array([[0, -1j], [1j, 0]])
        z_mat = np.array([[1, 0], [0, -1]])
        return self.global_phase * combinators.dot(after, interaction_matrix(z_mat, z), interaction_matrix(y_mat, y), interaction_matrix(x_mat, x), before)

    def _decompose_(self, qubits):
        from cirq import ops
        a, b = qubits
        return [ops.global_phase_operation(self.global_phase), ops.MatrixGate(self.single_qubit_operations_before[0]).on(a), ops.MatrixGate(self.single_qubit_operations_before[1]).on(b), np.exp(1j * ops.X(a) * ops.X(b) * self.interaction_coefficients[0]), np.exp(1j * ops.Y(a) * ops.Y(b) * self.interaction_coefficients[1]), np.exp(1j * ops.Z(a) * ops.Z(b) * self.interaction_coefficients[2]), ops.MatrixGate(self.single_qubit_operations_after[0]).on(a), ops.MatrixGate(self.single_qubit_operations_after[1]).on(b)]