from __future__ import annotations
import numpy as np
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.synthesis.two_qubit import TwoQubitBasisDecomposer
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes.synthesis import unitary_synthesis
from qiskit.transpiler.passes.utils import _block_to_matrix
from .collect_1q_runs import Collect1qRuns
from .collect_2q_blocks import Collect2qBlocks
def _block_qargs_to_indices(self, dag, block_qargs):
    """Map each qubit in block_qargs to its wire position among the block's wires.
        Args:
            block_qargs (list): list of qubits that a block acts on
            global_index_map (dict): mapping from each qubit in the
                circuit to its wire position within that circuit
        Returns:
            dict: mapping from qarg to position in block
        """
    block_indices = [dag.find_bit(q).index for q in block_qargs]
    ordered_block_indices = {bit: index for index, bit in enumerate(sorted(block_indices))}
    block_positions = {q: ordered_block_indices[dag.find_bit(q).index] for q in block_qargs}
    return block_positions