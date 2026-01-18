import pytest
import cirq
def test_value_equality_approximate():
    assert cirq.approx_eq(ApproxE(0.0), ApproxE(0.0), atol=0.1)
    assert cirq.approx_eq(ApproxE(0.0), ApproxE(0.2), atol=0.3)
    assert not cirq.approx_eq(ApproxE(0.0), ApproxE(0.2), atol=0.1)