import pytest
import numpy as np
import sympy
import cirq
def test_not_implemented_diagram():
    q = cirq.LineQubit.range(2)
    g = cirq.testing.SingleQubitGate()
    c = cirq.Circuit()
    c.append(cirq.ParallelGate(g, 2)(*q))
    assert 'cirq.testing.gate_features.SingleQubitGate ' in str(c)