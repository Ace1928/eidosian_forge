import time
import logging
from functools import singledispatchmethod
from itertools import zip_longest
from collections import defaultdict
import rustworkx
from qiskit.circuit import (
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.equivalence import Key, NodeData
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
class BasisSearchVisitor(rustworkx.visit.DijkstraVisitor):
    """Handles events emitted during `rustworkx.dijkstra_search`."""

    def __init__(self, graph, source_basis, target_basis):
        self.graph = graph
        self.target_basis = set(target_basis)
        self._source_gates_remain = set(source_basis)
        self._num_gates_remain_for_rule = {}
        save_index = -1
        for edata in self.graph.edges():
            if save_index == edata.index:
                continue
            self._num_gates_remain_for_rule[edata.index] = edata.num_gates
            save_index = edata.index
        self._basis_transforms = []
        self._predecessors = {}
        self._opt_cost_map = {}

    def discover_vertex(self, v, score):
        gate = self.graph[v].key
        self._source_gates_remain.discard(gate)
        self._opt_cost_map[gate] = score
        rule = self._predecessors.get(gate, None)
        if rule is not None:
            logger.debug('Gate %s generated using rule \n%s\n with total cost of %s.', gate.name, rule.circuit, score)
            self._basis_transforms.append((gate.name, gate.num_qubits, rule.params, rule.circuit))
        if not self._source_gates_remain:
            self._basis_transforms.reverse()
            raise StopIfBasisRewritable

    def examine_edge(self, edge):
        _, target, edata = edge
        if edata is None:
            return
        self._num_gates_remain_for_rule[edata.index] -= 1
        target = self.graph[target].key
        if self._num_gates_remain_for_rule[edata.index] > 0 or target in self.target_basis:
            raise rustworkx.visit.PruneSearch

    def edge_relaxed(self, edge):
        _, target, edata = edge
        if edata is not None:
            gate = self.graph[target].key
            self._predecessors[gate] = edata.rule

    def edge_cost(self, edge_data):
        """Returns the cost of an edge.

        This function computes the cost of this edge rule by summing
        the costs of all gates in the rule equivalence circuit. In the
        end, we need to subtract the cost of the source since `dijkstra`
        will later add it.
        """
        if edge_data is None:
            return 1
        cost_tot = 0
        for instruction in edge_data.rule.circuit:
            key = Key(name=instruction.operation.name, num_qubits=len(instruction.qubits))
            cost_tot += self._opt_cost_map[key]
        return cost_tot - self._opt_cost_map[edge_data.source]

    @property
    def basis_transforms(self):
        """Returns the gate basis transforms."""
        return self._basis_transforms