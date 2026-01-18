from typing import Optional, List, Hashable, TYPE_CHECKING
import abc
from cirq import circuits, ops, protocols, transformers
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers import merge_k_qubit_gates, merge_single_qubit_gates
def transformer_with_kwargs(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None) -> 'cirq.AbstractCircuit':
    return transformer(circuit, context=context, **kwargs)