import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_simulate_parameters_not_resolved():
    a = cirq.LineQubit(0)
    simulator = cirq.CliffordSimulator()
    circuit = cirq.Circuit(cirq.XPowGate(exponent=sympy.Symbol('a'))(a), cirq.measure(a))
    with pytest.raises(ValueError, match='symbols were not specified'):
        _ = simulator.simulate_sweep(circuit, cirq.ParamResolver({}))