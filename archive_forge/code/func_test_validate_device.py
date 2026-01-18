import numpy as np
import pytest
import sympy
import cirq
def test_validate_device():
    device = OnlyMeasurementsDevice()
    sampler = cirq.ZerosSampler(device)
    a, b, c = [cirq.NamedQubit(s) for s in ['a', 'b', 'c']]
    circuit = cirq.Circuit(cirq.measure(a), cirq.measure(b, c))
    _ = sampler.run_sweep(circuit, None, 3)
    circuit = cirq.Circuit(cirq.measure(a), cirq.X(b))
    with pytest.raises(ValueError, match='X\\(b\\) is not a measurement'):
        _ = sampler.run_sweep(circuit, None, 3)