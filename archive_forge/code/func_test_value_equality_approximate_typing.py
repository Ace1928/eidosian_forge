import pytest
import cirq
def test_value_equality_approximate_typing():
    assert not cirq.approx_eq(ApproxE(0.0), PeriodicF(0.0, 1.0), atol=0.1)
    assert cirq.approx_eq(ApproxEa(0.0), ApproxEb(0.0), atol=0.1)
    assert cirq.approx_eq(ApproxG(0.0), ApproxG(0.0), atol=0.1)
    assert not cirq.approx_eq(ApproxGa(0.0), ApproxGb(0.0), atol=0.1)
    assert not cirq.approx_eq(ApproxG(0.0), ApproxGb(0.0), atol=0.1)