from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def apply_map_func(op: 'cirq.Operation', idx: int) -> List['cirq.Operation']:
    if tags_to_ignore_set.intersection(op.tags):
        return [op]
    if deep and isinstance(op.untagged, circuits.CircuitOperation):
        op = op.untagged.replace(circuit=_map_operations_impl(op.untagged.circuit, map_func, deep=deep, raise_if_add_qubits=raise_if_add_qubits, tags_to_ignore=tags_to_ignore, wrap_in_circuit_op=wrap_in_circuit_op)).with_tags(*op.tags)
    mapped_ops = [*ops.flatten_to_ops(map_func(op, idx))]
    op_qubits = set(op.qubits)
    mapped_ops_qubits: Set['cirq.Qid'] = set()
    has_overlapping_ops = False
    for mapped_op in mapped_ops:
        if raise_if_add_qubits and (not op_qubits.issuperset(mapped_op.qubits)):
            raise ValueError(f'Mapped operations {mapped_ops} should act on a subset of qubits of the original operation {op}')
        if mapped_ops_qubits.intersection(mapped_op.qubits):
            has_overlapping_ops = True
        mapped_ops_qubits = mapped_ops_qubits.union(mapped_op.qubits)
    if wrap_in_circuit_op and has_overlapping_ops:
        mapped_ops = [circuits.CircuitOperation(circuits.FrozenCircuit(mapped_ops)).with_tags(MAPPED_CIRCUIT_OP_TAG)]
    return mapped_ops