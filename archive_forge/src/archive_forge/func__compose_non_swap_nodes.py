from __future__ import annotations
from collections import defaultdict
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass, Layout, TranspilerError
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.swap_strategy import SwapStrategy
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
def _compose_non_swap_nodes(self, accumulator: DAGCircuit, layout: Layout, new_dag: DAGCircuit) -> DAGCircuit:
    """Add all the non-swap strategy nodes that we have accumulated up to now.

        This method also resets the node accumulator to an empty dag.

        Args:
            layout: The current layout that keeps track of the swaps.
            new_dag: The new dag that we are building up.
            accumulator: A DAG to keep track of nodes that do not decompose
                using swap strategies.

        Returns:
            A new accumulator with the same registers as ``new_dag``.
        """
    order = layout.reorder_bits(new_dag.qubits)
    order_bits: list[int | None] = [None] * len(layout)
    for idx, val in enumerate(order):
        order_bits[val] = idx
    new_dag.compose(accumulator, qubits=order_bits)
    return new_dag.copy_empty_like()