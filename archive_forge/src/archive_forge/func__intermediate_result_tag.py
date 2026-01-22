from typing import Optional, List, Hashable, TYPE_CHECKING
import abc
from cirq import circuits, ops, protocols, transformers
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers import merge_k_qubit_gates, merge_single_qubit_gates
@property
def _intermediate_result_tag(self) -> Hashable:
    """A tag used to identify intermediate compilation results."""
    return '_default_merged_k_qubit_unitaries'