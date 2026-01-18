import pytest
import cirq
def test_value_equality_approximate_specialized():
    assert PeriodicF(1, 4) != PeriodicF(5, 4)
    assert cirq.approx_eq(PeriodicF(1, 4), PeriodicF(5, 4), atol=0.1)
    assert not cirq.approx_eq(PeriodicF(1, 4), PeriodicF(6, 4), atol=0.1)