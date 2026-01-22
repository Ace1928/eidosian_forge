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
def _extract_basis_target(self, dag, qarg_indices, source_basis=None, qargs_local_source_basis=None):
    if source_basis is None:
        source_basis = set()
    if qargs_local_source_basis is None:
        qargs_local_source_basis = defaultdict(set)
    for node in dag.op_nodes():
        qargs = tuple((qarg_indices[bit] for bit in node.qargs))
        if dag.has_calibration_for(node) or len(node.qargs) < self._min_qubits:
            continue
        if qargs in self._qargs_with_non_global_operation or any((frozenset(qargs).issuperset(incomplete_qargs) for incomplete_qargs in self._qargs_with_non_global_operation)):
            qargs_local_source_basis[frozenset(qargs)].add((node.name, node.op.num_qubits))
        else:
            source_basis.add((node.name, node.op.num_qubits))
        if isinstance(node.op, ControlFlowOp):
            for block in node.op.blocks:
                block_dag = circuit_to_dag(block)
                source_basis, qargs_local_source_basis = self._extract_basis_target(block_dag, {inner: qarg_indices[outer] for inner, outer in zip(block.qubits, node.qargs)}, source_basis=source_basis, qargs_local_source_basis=qargs_local_source_basis)
    return (source_basis, qargs_local_source_basis)