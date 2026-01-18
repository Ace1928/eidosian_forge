import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_run_no_repetitions():
    q0 = cirq.LineQubit(0)
    simulator = ccq.mps_simulator.MPSSimulator()
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    result = simulator.run(circuit, repetitions=0)
    assert len(result.measurements['q(0)']) == 0