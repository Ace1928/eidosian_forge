from typing import Any, Dict, Optional, Sequence
import cirq
def test_swap_does_not_merge():
    state = create_container(qs2)
    old_q0 = state[q0]
    old_q1 = state[q1]
    state.apply_operation(cirq.SWAP(q0, q1))
    assert len(set(state.values())) == 3
    assert state[q0] is not old_q0
    assert state[q1] is old_q0
    assert state[q1] is not old_q1
    assert state[q0] is old_q1
    assert state[q0].qubits == (q0,)
    assert state[q1].qubits == (q1,)