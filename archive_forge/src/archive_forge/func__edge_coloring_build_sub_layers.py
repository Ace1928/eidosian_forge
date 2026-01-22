from __future__ import annotations
from collections import defaultdict
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass, Layout, TranspilerError
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.swap_strategy import SwapStrategy
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
def _edge_coloring_build_sub_layers(self, current_layer: dict[tuple[int, int], Gate]) -> list[dict[tuple[int, int], Gate]]:
    """The edge coloring method of building sub-layers of commuting gates."""
    sub_layers: list[dict[tuple[int, int], Gate]] = [{} for _ in set(self._edge_coloring.values())]
    for edge, gate in current_layer.items():
        color = self._edge_coloring[edge]
        sub_layers[color][edge] = gate
    return sub_layers