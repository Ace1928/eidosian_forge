from typing import Optional, TYPE_CHECKING
from cirq import circuits, ops, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers import transformer_api, transformer_primitives, merge_k_qubit_gates
Merges adjacent moments with only 1-qubit rotations to a single moment with PhasedXZ gates.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        atol: Absolute tolerance to angle error. Larger values allow more negligible gates to be
            dropped, smaller values increase accuracy.

    Returns:
        Copy of the transformed input circuit.
    