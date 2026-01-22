from __future__ import annotations
from collections import defaultdict
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass, Layout, TranspilerError
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.swap_strategy import SwapStrategy
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
def _position_in_cmap(self, dag: DAGCircuit, j: int, k: int, layout: Layout) -> tuple[int, ...]:
    """A helper function to track the movement of virtual qubits through the swaps.

        Args:
            j: The index of decision variable j (i.e. virtual qubit).
            k: The index of decision variable k (i.e. virtual qubit).
            layout: The current layout that takes into account previous swap gates.

        Returns:
            The position in the coupling map of the virtual qubits j and k as a tuple.
        """
    bit0 = dag.find_bit(layout.get_physical_bits()[j]).index
    bit1 = dag.find_bit(layout.get_physical_bits()[k]).index
    return (bit0, bit1)