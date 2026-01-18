import pytest
import cirq
@pytest.mark.parametrize('val', (NoMethod(), 'text', object(), ReturnsNotImplemented()))
def test_powerless(val):
    assert cirq.pow(val, 5, None) is None
    assert cirq.pow(val, 2, NotImplemented) is NotImplemented
    assert cirq.pow(val, 1, None) is None