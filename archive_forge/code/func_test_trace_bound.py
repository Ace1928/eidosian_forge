import itertools
import numpy as np
import pytest
import sympy
import cirq
def test_trace_bound():
    assert cirq.trace_distance_bound(cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.001)) < 0.01
    assert cirq.trace_distance_bound(cirq.PhasedXPowGate(phase_exponent=0.25, exponent=sympy.Symbol('a'))) >= 1