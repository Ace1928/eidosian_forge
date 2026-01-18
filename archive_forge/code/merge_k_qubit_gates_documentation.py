from typing import cast, Optional, Callable, TYPE_CHECKING
from cirq import ops, protocols, circuits
from cirq.transformers import transformer_api, transformer_primitives
Merges connected components of unitary operations, acting on <= k qubits.

    Uses rewriter to convert a connected component of unitary operations acting on <= k-qubits
    into a more desirable form. If not specified, connected components are replaced by a single
    `cirq.MatrixGate` containing unitary matrix of the merged component.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        k: Connected components of unitary operations acting on <= k qubits are merged.
        rewriter: Callable type that takes a `cirq.CircuitOperation`, encapsulating a connected
            component of unitary operations acting on <= k qubits, and produces a `cirq.OP_TREE`.
            Specifies how to merge the connected component into a more desirable form.

    Returns:
        Copy of the transformed input circuit.

    Raises:
        ValueError: If k <= 0
    