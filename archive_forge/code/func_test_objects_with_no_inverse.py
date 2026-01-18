import pytest
import cirq
@pytest.mark.parametrize('val', (NoMethod(), 'text', object(), ReturnsNotImplemented(), [NoMethod(), 5]))
def test_objects_with_no_inverse(val):
    with pytest.raises(TypeError, match="isn't invertible"):
        _ = cirq.inverse(val)
    assert cirq.inverse(val, None) is None
    assert cirq.inverse(val, NotImplemented) is NotImplemented
    assert cirq.inverse(val, 5) == 5