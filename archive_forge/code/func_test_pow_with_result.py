import pytest
import cirq
@pytest.mark.parametrize('val,exponent,out', ((ReturnsExponent(), 2, 2), (1, 2, 1), (2, 3, 8)))
def test_pow_with_result(val, exponent, out):
    assert cirq.pow(val, exponent) == cirq.pow(val, exponent, default=None) == val ** exponent == out