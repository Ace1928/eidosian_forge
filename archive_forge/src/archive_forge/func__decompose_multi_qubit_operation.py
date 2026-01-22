from typing import Optional, List, Hashable, TYPE_CHECKING
import abc
from cirq import circuits, ops, protocols, transformers
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers import merge_k_qubit_gates, merge_single_qubit_gates
def _decompose_multi_qubit_operation(self, op: 'cirq.Operation', moment_idx: int) -> DecomposeResult:
    """Decomposes operations acting on more than 2 qubits using gates from this gateset.

        Args:
            op: A multi qubit (>2q) operation.
            moment_idx: Index of the moment in which operation `op` occurs.

        Returns:
            A `cirq.OP_TREE` implementing `op` using gates from this gateset OR
            None or NotImplemented if decomposition of `op` is unknown.
        """
    return NotImplemented