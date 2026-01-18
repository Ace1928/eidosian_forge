import pytest
import cirq
def test_q_invalid() -> None:
    with pytest.raises(ValueError):
        cirq.q([1, 2, 3])
    with pytest.raises(ValueError):
        cirq.q(1, 'foo')