import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_separated_states_str_does_not_merge():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0), cirq.measure(q1), cirq.X(q0))
    result = cirq.DensityMatrixSimulator().simulate(circuit)
    assert str(result) == 'measurements: q(0)=0 q(1)=0\n\nqubits: (cirq.LineQubit(0),)\nfinal density matrix:\n[[0.+0.j 0.+0.j]\n [0.+0.j 1.+0.j]]\n\nqubits: (cirq.LineQubit(1),)\nfinal density matrix:\n[[1.+0.j 0.+0.j]\n [0.+0.j 0.+0.j]]\n\nphase:\nfinal density matrix:\n[[1.+0.j]]'