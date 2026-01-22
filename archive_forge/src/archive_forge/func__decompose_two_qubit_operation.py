import itertools
from typing import cast, Any, Dict, List, Optional, Sequence
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq_google import ops
from cirq_google.transformers.analytical_decompositions import two_qubit_to_sycamore
def _decompose_two_qubit_operation(self, op: cirq.Operation, _) -> DecomposeResult:
    if not cirq.has_unitary(op):
        return NotImplemented
    known_decomp = two_qubit_to_sycamore.known_2q_op_to_sycamore_operations(op)
    if known_decomp is not None:
        return known_decomp
    if self.tabulation is not None:
        return two_qubit_to_sycamore._decompose_arbitrary_into_syc_tabulation(op, self.tabulation)
    return two_qubit_to_sycamore.two_qubit_matrix_to_sycamore_operations(op.qubits[0], op.qubits[1], cirq.unitary(op))