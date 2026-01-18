import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_scaled_unitary_consistency():
    a, b = cirq.LineQubit.range(2)
    cirq.testing.assert_implements_consistent_protocols(2 * cirq.X(a) * cirq.Y(b))
    cirq.testing.assert_implements_consistent_protocols(1j * cirq.X(a) * cirq.Y(b))