from typing import Tuple
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
Decompose the SparsePauliOp into two-qubit.

        Args:
            dag: The dag needed to get access to qubits.
            op: The operator with all the Pauli terms we need to apply.

        Returns:
            A dag made of two-qubit :class:`.PauliEvolutionGate`.
        