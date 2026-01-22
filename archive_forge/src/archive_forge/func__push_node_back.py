from __future__ import annotations
from collections.abc import Generator
from qiskit.circuit.gate import Gate
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGOutNode
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
def _push_node_back(self, dag: DAGCircuit, node: DAGOpNode):
    """Update the start time of the current node to satisfy alignment constraints.
        Immediate successors are pushed back to avoid overlap and will be processed later.

        .. note::

            This logic assumes the all bits in the qregs and cregs synchronously start and end,
            i.e. occupy the same time slot, but qregs and cregs can take
            different time slot due to classical I/O latencies.

        Args:
            dag: DAG circuit to be rescheduled with constraints.
            node: Current node.
        """
    node_start_time = self.property_set['node_start_time']
    conditional_latency = self.property_set.get('conditional_latency', 0)
    clbit_write_latency = self.property_set.get('clbit_write_latency', 0)
    if isinstance(node.op, Gate):
        alignment = self.pulse_align
    elif isinstance(node.op, Measure):
        alignment = self.acquire_align
    elif isinstance(node.op, Delay) or getattr(node.op, '_directive', False):
        alignment = None
    else:
        raise TranspilerError(f'Unknown operation type for {repr(node)}.')
    this_t0 = node_start_time[node]
    if alignment is not None:
        misalignment = node_start_time[node] % alignment
        if misalignment != 0:
            shift = max(0, alignment - misalignment)
        else:
            shift = 0
        this_t0 += shift
        node_start_time[node] = this_t0
    new_t1q = this_t0 + node.op.duration
    this_qubits = set(node.qargs)
    if isinstance(node.op, Measure):
        new_t1c = new_t1q
        this_clbits = set(node.cargs)
    elif node.op.condition_bits:
        new_t1c = this_t0
        this_clbits = set(node.op.condition_bits)
    else:
        new_t1c = None
        this_clbits = set()
    for next_node in self._get_next_gate(dag, node):
        next_t0q = node_start_time[next_node]
        next_qubits = set(next_node.qargs)
        if isinstance(next_node.op, Measure):
            next_t0c = next_t0q + clbit_write_latency
            next_clbits = set(next_node.cargs)
        elif next_node.op.condition_bits:
            next_t0c = next_t0q - conditional_latency
            next_clbits = set(next_node.op.condition_bits)
        else:
            next_t0c = None
            next_clbits = set()
        if any(this_qubits & next_qubits):
            qreg_overlap = new_t1q - next_t0q
        else:
            qreg_overlap = 0
        if any(this_clbits & next_clbits):
            creg_overlap = new_t1c - next_t0c
        else:
            creg_overlap = 0
        overlap = max(qreg_overlap, creg_overlap)
        node_start_time[next_node] = node_start_time[next_node] + overlap