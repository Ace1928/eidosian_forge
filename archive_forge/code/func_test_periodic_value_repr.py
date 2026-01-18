import pytest
import sympy
import cirq
@pytest.mark.parametrize('val', [cirq.PeriodicValue(0.4, 1.0), cirq.PeriodicValue(0.0, 2.0), cirq.PeriodicValue(1.0, 3), cirq.PeriodicValue(-2.1, 3.0), cirq.PeriodicValue(sympy.Symbol('v'), sympy.Symbol('p')), cirq.PeriodicValue(2.0, sympy.Symbol('p')), cirq.PeriodicValue(sympy.Symbol('v'), 3)])
def test_periodic_value_repr(val):
    cirq.testing.assert_equivalent_repr(val)