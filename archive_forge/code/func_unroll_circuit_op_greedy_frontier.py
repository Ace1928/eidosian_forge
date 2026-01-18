from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def unroll_circuit_op_greedy_frontier(circuit: CIRCUIT_TYPE, *, deep: bool=False, tags_to_check: Optional[Sequence[Hashable]]=(MAPPED_CIRCUIT_OP_TAG,)) -> CIRCUIT_TYPE:
    """Unrolls (tagged) `cirq.CircuitOperation`s by inserting operations inline at qubit frontier.

    Each matching `cirq.CircuitOperation` is replaced by inserting underlying operations using the
    `circuit.insert_at_frontier` method. The greedy approach attempts to reuse any available space
    in existing moments on the right of circuit_op before inserting new moments.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.
        tags_to_check: If specified, only circuit operations tagged with one of the `tags_to_check`
            are unrolled.

    Returns:
        Copy of input circuit with (Tagged) CircuitOperation's expanded inline at qubit frontier.
    """
    unrolled_circuit = circuit.unfreeze(copy=True)
    frontier: Dict['cirq.Qid', int] = defaultdict(lambda: 0)
    idx = 0
    while idx < len(unrolled_circuit):
        for op in unrolled_circuit[idx].operations:
            if not isinstance(op.untagged, circuits.CircuitOperation):
                continue
            if any((frontier[q] > idx for q in op.qubits)):
                continue
            op_untagged = op.untagged
            if deep:
                op_untagged = op_untagged.replace(circuit=unroll_circuit_op_greedy_frontier(op_untagged.circuit, deep=deep, tags_to_check=tags_to_check))
            if tags_to_check is None or set(tags_to_check).intersection(op.tags):
                unrolled_circuit.clear_operations_touching(op.qubits, [idx])
                frontier = unrolled_circuit.insert_at_frontier(op_untagged.mapped_circuit().all_operations(), idx, frontier)
            elif deep:
                unrolled_circuit.batch_replace([(idx, op, op_untagged.with_tags(*op.tags))])
        idx += 1
    return _to_target_circuit_type(unrolled_circuit, circuit)