from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def unroll_circuit_op_greedy_earliest(circuit: CIRCUIT_TYPE, *, deep: bool=False, tags_to_check: Optional[Sequence[Hashable]]=(MAPPED_CIRCUIT_OP_TAG,)) -> CIRCUIT_TYPE:
    """Unrolls (tagged) `cirq.CircuitOperation`s by inserting operations using EARLIEST strategy.

    Each matching `cirq.CircuitOperation` is replaced by inserting underlying operations using the
    `cirq.InsertStrategy.EARLIEST` strategy. The greedy approach attempts to minimize circuit depth
    of the resulting circuit.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.
        tags_to_check: If specified, only circuit operations tagged with one of the `tags_to_check`
            are unrolled.

    Returns:
        Copy of input circuit with (Tagged) CircuitOperation's expanded using EARLIEST strategy.
    """
    batch_replace = []
    batch_remove = []
    batch_insert = []
    for i, op in circuit.findall_operations(lambda o: isinstance(o.untagged, circuits.CircuitOperation)):
        op_untagged = cast(circuits.CircuitOperation, op.untagged)
        if deep:
            op_untagged = op_untagged.replace(circuit=unroll_circuit_op_greedy_earliest(op_untagged.circuit, deep=deep, tags_to_check=tags_to_check))
        if tags_to_check is None or set(tags_to_check).intersection(op.tags):
            batch_remove.append((i, op))
            batch_insert.append((i, op_untagged.mapped_circuit().all_operations()))
        elif deep:
            batch_replace.append((i, op, op_untagged.with_tags(*op.tags)))
    unrolled_circuit = circuit.unfreeze(copy=True)
    unrolled_circuit.batch_replace(batch_replace)
    unrolled_circuit.batch_remove(batch_remove)
    unrolled_circuit.batch_insert(batch_insert)
    return _to_target_circuit_type(unrolled_circuit, circuit)