import re
import numpy as np
import pytest
import sympy
import cirq
def test_protocols_and_repr():
    cirq.testing.assert_implements_consistent_protocols(cirq.MatrixGate(np.diag([1, 1j, 1, -1])))
    cirq.testing.assert_implements_consistent_protocols(cirq.MatrixGate(np.diag([1, 1j, -1]), qid_shape=(3,)))