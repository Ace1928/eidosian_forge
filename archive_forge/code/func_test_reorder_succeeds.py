from typing import Any, Dict, Optional, Sequence
import cirq
def test_reorder_succeeds():
    state = create_container(qs2, False)
    reordered = state[q0].transpose_to_qubit_order([q1, q0])
    assert reordered.qubits == (q1, q0)