from typing import Any, FrozenSet, Sequence
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
def _try_decompose_from_known_decompositions(val: Any, context: cirq.DecompositionContext) -> DecomposeResult:
    """Returns a flattened decomposition of the object into operations, if possible.

    Args:
        val: The object to decompose.
        context: Decomposition context storing common configurable options for `cirq.decompose`.

    Returns:
        A flattened decomposition of `val` if it's a gate or operation with a known decomposition.
    """
    if not isinstance(val, (cirq.Gate, cirq.Operation)):
        return None
    qubits = cirq.LineQid.for_gate(val) if isinstance(val, cirq.Gate) else val.qubits
    known_decompositions = [(_FREDKIN_GATESET, _fredkin)]
    classical_controls: FrozenSet[cirq.Condition] = frozenset()
    if isinstance(val, cirq.ClassicallyControlledOperation):
        classical_controls = val.classical_controls
        val = val.without_classical_controls()
    decomposition = None
    for gateset, decomposer in known_decompositions:
        if val in gateset:
            decomposition = cirq.flatten_to_ops(decomposer(qubits, context))
            break
    return tuple((op.with_classical_controls(*classical_controls) for op in decomposition)) if decomposition else None