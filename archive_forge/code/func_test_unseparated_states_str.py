import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_unseparated_states_str():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0), cirq.measure(q1), cirq.X(q0))
    result = cirq.DensityMatrixSimulator(split_untangled_states=False).simulate(circuit)
    assert str(result) == 'measurements: q(0)=0 q(1)=0\n\nqubits: (cirq.LineQubit(0), cirq.LineQubit(1))\nfinal density matrix:\n[[0.+0.j 0.+0.j 0.+0.j 0.+0.j]\n [0.+0.j 0.+0.j 0.+0.j 0.+0.j]\n [0.+0.j 0.+0.j 1.+0.j 0.+0.j]\n [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]'