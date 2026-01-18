import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_simulates_composite():
    c = cirq.Circuit(MultiHTestGate().on(*cirq.LineQubit.range(2)))
    expected = np.array([0.5] * 4)
    np.testing.assert_allclose(c.final_state_vector(ignore_terminal_measurements=False, dtype=np.complex64), expected)
    np.testing.assert_allclose(cirq.Simulator().simulate(c).state_vector(), expected)