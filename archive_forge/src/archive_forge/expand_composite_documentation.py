from typing import Callable, Optional, TYPE_CHECKING
from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
A transformer that expands composite operations via `cirq.decompose`.

    For each operation in the circuit, this pass examines if the operation can
    be decomposed. If it can be, the operation is cleared out and replaced
    with its decomposition using a fixed insertion strategy.

    Transformation is applied using `cirq.map_operations_and_unroll`, which preserves the
    moment structure of the input circuit.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.
          no_decomp: A predicate that determines whether an operation should
                be decomposed or not. Defaults to decomposing everything.
    Returns:
          Copy of the transformed input circuit.
    