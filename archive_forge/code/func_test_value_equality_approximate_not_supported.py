import pytest
import cirq
def test_value_equality_approximate_not_supported():
    assert not cirq.approx_eq(BasicC(0.0), BasicC(0.1), atol=0.2)