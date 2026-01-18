import pytest
import cirq
def test_value_equality_unhashable():
    with pytest.raises(TypeError, match='unhashable'):
        _ = {UnhashableC(1): 4}
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(UnhashableC(1), UnhashableC(1), UnhashableCa(1), UnhashableCb(1))
    eq.add_equality_group(UnhashableC(2))
    eq.add_equality_group(UnhashableD(1))