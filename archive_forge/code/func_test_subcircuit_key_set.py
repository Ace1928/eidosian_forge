import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_subcircuit_key_set(sim):
    q0, q1 = cirq.LineQubit.range(2)
    inner = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key='c'), cirq.X(q1).with_classical_controls('c'), cirq.measure(q1, key='b'))
    circuit = cirq.Circuit(cirq.CircuitOperation(inner.freeze(), repetitions=4, measurement_key_map={'c': 'a'}))
    result = sim.run(circuit)
    assert result.measurements['0:a'] == 1
    assert result.measurements['0:b'] == 1
    assert result.measurements['1:a'] == 0
    assert result.measurements['1:b'] == 1
    assert result.measurements['2:a'] == 1
    assert result.measurements['2:b'] == 0
    assert result.measurements['3:a'] == 0
    assert result.measurements['3:b'] == 0