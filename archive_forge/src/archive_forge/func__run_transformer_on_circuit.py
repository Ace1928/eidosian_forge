import dataclasses
import inspect
import enum
import functools
import textwrap
from typing import (
from typing_extensions import Protocol
from cirq import circuits
def _run_transformer_on_circuit(add_deep_support: bool, func: TRANSFORMER, circuit: 'cirq.AbstractCircuit', extracted_context: Optional[TransformerContext], **kwargs) -> 'cirq.AbstractCircuit':
    mutable_circuit = None
    if extracted_context and extracted_context.deep and add_deep_support:
        batch_replace = []
        for i, op in circuit.findall_operations(lambda o: isinstance(o.untagged, circuits.CircuitOperation)):
            op_untagged = cast(circuits.CircuitOperation, op.untagged)
            if not set(op.tags).isdisjoint(extracted_context.tags_to_ignore):
                continue
            op_untagged = op_untagged.replace(circuit=_run_transformer_on_circuit(add_deep_support, func, op_untagged.circuit, extracted_context, **kwargs).freeze())
            batch_replace.append((i, op, op_untagged.with_tags(*op.tags)))
        mutable_circuit = circuit.unfreeze(copy=True)
        mutable_circuit.batch_replace(batch_replace)
    return func(mutable_circuit if mutable_circuit else circuit, **kwargs)