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
def apply_translation(dag, wire_map):
    dag_updated = False
    for node in dag.op_nodes():
        node_qargs = tuple((wire_map[bit] for bit in node.qargs))
        qubit_set = frozenset(node_qargs)
        if node.name in target_basis or len(node.qargs) < self._min_qubits:
            if isinstance(node.op, ControlFlowOp):
                flow_blocks = []
                for block in node.op.blocks:
                    dag_block = circuit_to_dag(block)
                    dag_updated = apply_translation(dag_block, {inner: wire_map[outer] for inner, outer in zip(block.qubits, node.qargs)})
                    if dag_updated:
                        flow_circ_block = dag_to_circuit(dag_block)
                    else:
                        flow_circ_block = block
                    flow_blocks.append(flow_circ_block)
                node.op = node.op.replace_blocks(flow_blocks)
            continue
        if node_qargs in self._qargs_with_non_global_operation and node.name in self._qargs_with_non_global_operation[node_qargs]:
            continue
        if dag.has_calibration_for(node):
            continue
        if qubit_set in extra_instr_map:
            self._replace_node(dag, node, extra_instr_map[qubit_set])
        elif (node.op.name, node.op.num_qubits) in instr_map:
            self._replace_node(dag, node, instr_map)
        else:
            raise TranspilerError(f'BasisTranslator did not map {node.name}.')
        dag_updated = True
    return dag_updated