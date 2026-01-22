from __future__ import annotations
from collections.abc import Sequence
import math
import numpy as np
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
class GateSequence:
    """A class implementing a sequence of gates.

    This class stores the sequence of gates along with the unitary they implement.
    """

    def __init__(self, gates: Sequence[Gate]=()) -> None:
        """Create a new sequence of gates.

        Args:
            gates: The gates in the sequence. The default is [].
        """
        self.gates = list(gates)
        self.matrices = [np.asarray(gate, dtype=np.complex128) for gate in gates]
        self.labels = [gate.name for gate in gates]
        u2_matrix = np.identity(2)
        for matrix in self.matrices:
            u2_matrix = matrix.dot(u2_matrix)
        su2_matrix, global_phase = _convert_u2_to_su2(u2_matrix)
        so3_matrix = _convert_su2_to_so3(su2_matrix)
        self._eulers = None
        self.name = ' '.join(self.labels)
        self.global_phase = global_phase
        self.product = so3_matrix
        self.product_su2 = su2_matrix

    def remove_cancelling_pair(self, indices: Sequence[int]) -> None:
        """Remove a pair of indices that cancel each other and *do not* change the matrices."""
        for index in list(indices[::-1]):
            self.gates.pop(index)
            self.labels.pop(index)
        self.name = ' '.join(self.labels)

    def __eq__(self, other: 'GateSequence') -> bool:
        """Check if this GateSequence is the same as the other GateSequence.

        Args:
            other: The GateSequence that will be compared to ``self``.

        Returns:
            True if ``other`` is equivalent to ``self``, false otherwise.

        """
        if not len(self.gates) == len(other.gates):
            return False
        for gate1, gate2 in zip(self.gates, other.gates):
            if gate1 != gate2:
                return False
        if self.global_phase != other.global_phase:
            return False
        return True

    def to_circuit(self):
        """Convert to a circuit.

        If no gates set but the product is not the identity, returns a circuit with a
        unitary operation to implement the matrix.
        """
        if len(self.gates) == 0 and (not np.allclose(self.product, np.identity(3))):
            circuit = QuantumCircuit(1, global_phase=self.global_phase)
            su2 = _convert_so3_to_su2(self.product)
            circuit.unitary(su2, [0])
            return circuit
        circuit = QuantumCircuit(1, global_phase=self.global_phase)
        for gate in self.gates:
            circuit.append(gate, [0])
        return circuit

    def to_dag(self):
        """Convert to a :class:`.DAGCircuit`.

        If no gates set but the product is not the identity, returns a circuit with a
        unitary operation to implement the matrix.
        """
        from qiskit.dagcircuit import DAGCircuit
        qreg = (Qubit(),)
        dag = DAGCircuit()
        dag.add_qubits(qreg)
        if len(self.gates) == 0 and (not np.allclose(self.product, np.identity(3))):
            su2 = _convert_so3_to_su2(self.product)
            dag.apply_operation_back(UnitaryGate(su2), qreg, check=False)
            return dag
        dag.global_phase = self.global_phase
        for gate in self.gates:
            dag.apply_operation_back(gate, qreg, check=False)
        return dag

    def append(self, gate: Gate) -> 'GateSequence':
        """Append gate to the sequence of gates.

        Args:
            gate: The gate to be appended.

        Returns:
            GateSequence with ``gate`` appended.
        """
        self._eulers = None
        matrix = np.array(gate, dtype=np.complex128)
        su2, phase = _convert_u2_to_su2(matrix)
        so3 = _convert_su2_to_so3(su2)
        self.product = so3.dot(self.product)
        self.product_su2 = su2.dot(self.product_su2)
        self.global_phase = self.global_phase + phase
        self.gates.append(gate)
        if len(self.labels) > 0:
            self.name += f' {gate.name}'
        else:
            self.name = gate.name
        self.labels.append(gate.name)
        self.matrices.append(matrix)
        return self

    def adjoint(self) -> 'GateSequence':
        """Get the complex conjugate."""
        adjoint = GateSequence()
        adjoint.gates = [gate.inverse() for gate in reversed(self.gates)]
        adjoint.labels = [inv.name for inv in adjoint.gates]
        adjoint.name = ' '.join(adjoint.labels)
        adjoint.product = np.conj(self.product).T
        adjoint.product_su2 = np.conj(self.product_su2).T
        adjoint.global_phase = -self.global_phase
        return adjoint

    def copy(self) -> 'GateSequence':
        """Create copy of the sequence of gates.

        Returns:
            A new ``GateSequence`` containing copy of list of gates.

        """
        out = type(self).__new__(type(self))
        out.labels = self.labels.copy()
        out.gates = self.gates.copy()
        out.matrices = self.matrices.copy()
        out.global_phase = self.global_phase
        out.product = self.product.copy()
        out.product_su2 = self.product_su2.copy()
        out.name = self.name
        out._eulers = self._eulers
        return out

    def __len__(self) -> int:
        """Return length of sequence of gates.

        Returns:
            Length of list containing gates.
        """
        return len(self.gates)

    def __getitem__(self, index: int) -> Gate:
        """Returns the gate at ``index`` from the list of gates.

        Args
            index: Index of gate in list that will be returned.

        Returns:
            The gate at ``index`` in the list of gates.
        """
        return self.gates[index]

    def __repr__(self) -> str:
        """Return string representation of this object.

        Returns:
            Representation of this sequence of gates.
        """
        out = '['
        for gate in self.gates:
            out += gate.name
            out += ', '
        out += ']'
        out += ', product: '
        out += str(self.product)
        return out

    def __str__(self) -> str:
        """Return string representation of this object.

        Returns:
            Representation of this sequence of gates.
        """
        out = '['
        for gate in self.gates:
            out += gate.name
            out += ', '
        out += ']'
        out += ', product: \n'
        out += str(self.product)
        return out

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'GateSequence':
        """Initialize the gate sequence from a matrix, without a gate sequence.

        Args:
            matrix: The matrix, can be SU(2) or SO(3).

        Returns:
            A ``GateSequence`` initialized from the input matrix.

        Raises:
            ValueError: If the matrix has an invalid shape.
        """
        instance = cls()
        if matrix.shape == (2, 2):
            instance.product = _convert_su2_to_so3(matrix)
        elif matrix.shape == (3, 3):
            instance.product = matrix
        else:
            raise ValueError(f'Matrix must have shape (3, 3) or (2, 2) but has {matrix.shape}.')
        instance.gates = []
        return instance

    def dot(self, other: 'GateSequence') -> 'GateSequence':
        """Compute the dot-product with another gate sequence.

        Args:
            other: The other gate sequence.

        Returns:
            The dot-product as gate sequence.
        """
        composed = GateSequence()
        composed.gates = other.gates + self.gates
        composed.labels = other.labels + self.labels
        composed.name = ' '.join(composed.labels)
        composed.product = np.dot(self.product, other.product)
        composed.global_phase = self.global_phase + other.global_phase
        return composed