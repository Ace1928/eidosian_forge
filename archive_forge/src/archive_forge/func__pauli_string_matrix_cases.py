import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def _pauli_string_matrix_cases():
    q0, q1, q2 = cirq.LineQubit.range(3)
    return ((cirq.X(q0) * 2, None, np.array([[0, 2], [2, 0]])), (cirq.X(q0) * cirq.Y(q1), (q0,), np.array([[0, 1], [1, 0]])), (cirq.X(q0) * cirq.Y(q1), (q1,), np.array([[0, -1j], [1j, 0]])), (cirq.X(q0) * cirq.Y(q1), None, np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]])), (cirq.X(q0) * cirq.Y(q1), (q0, q1), np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]])), (cirq.X(q0) * cirq.Y(q1), (q1, q0), np.array([[0, 0, 0, -1j], [0, 0, -1j, 0], [0, 1j, 0, 0], [1j, 0, 0, 0]])), (cirq.X(q0) * cirq.Y(q1), (q2,), np.eye(2)), (cirq.X(q0) * cirq.Y(q1), (q2, q1), np.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])), (cirq.X(q0) * cirq.Y(q1), (q2, q0, q1), np.array([[0, 0, 0, -1j, 0, 0, 0, 0], [0, 0, 1j, 0, 0, 0, 0, 0], [0, -1j, 0, 0, 0, 0, 0, 0], [1j, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1j], [0, 0, 0, 0, 0, 0, 1j, 0], [0, 0, 0, 0, 0, -1j, 0, 0], [0, 0, 0, 0, 1j, 0, 0, 0]])))