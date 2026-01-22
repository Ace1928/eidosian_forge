from __future__ import annotations
from collections import defaultdict
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass, Layout, TranspilerError
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.swap_strategy import SwapStrategy
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
def _make_op_layers(self, dag: DAGCircuit, op: Commuting2qBlock, layout: Layout, swap_strategy: SwapStrategy) -> dict[int, dict[tuple, Gate]]:
    """Creates layers of two-qubit gates based on the distance in the swap strategy."""
    gate_layers: dict[int, dict[tuple, Gate]] = defaultdict(dict)
    for node in op.node_block:
        edge = (dag.find_bit(node.qargs[0]).index, dag.find_bit(node.qargs[1]).index)
        bit0 = layout.get_virtual_bits()[dag.qubits[edge[0]]]
        bit1 = layout.get_virtual_bits()[dag.qubits[edge[1]]]
        distance = swap_strategy.distance_matrix[bit0, bit1]
        gate_layers[distance][edge] = node.op
    return gate_layers