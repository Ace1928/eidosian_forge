from typing import Any, Dict, Optional, Sequence
import cirq
def test_reset_causes_split():
    state = create_container(qs2)
    state.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(state.values())) == 2
    state.apply_operation(cirq.reset(q0))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]
    assert state[q0] is not state[None]