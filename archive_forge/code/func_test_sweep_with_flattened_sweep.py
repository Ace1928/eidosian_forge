from typing import Iterator
import pytest
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2
def test_sweep_with_flattened_sweep():
    q = cirq.GridQubit(0, 0)
    circuit = cirq.Circuit(cirq.PhasedXPowGate(exponent=sympy.Symbol('t') / 4 + 0.5, phase_exponent=sympy.Symbol('t') / 2 + 0.1, global_shift=0.0)(q), cirq.measure(q, key='m'))
    param_sweep1 = cirq.Linspace('t', start=0, stop=1, length=20)
    _, param_sweep2 = cirq.flatten_with_sweep(circuit, param_sweep1)
    assert v2.sweep_to_proto(param_sweep2) is not None