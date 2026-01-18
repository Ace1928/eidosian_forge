import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_transform_qubits():
    a, b, c = cirq.LineQubit.range(3)
    p = cirq.X(a) * cirq.Z(b)
    p2 = cirq.X(b) * cirq.Z(c)
    m = p.mutable_copy()
    m2 = m.transform_qubits(lambda q: q + 1)
    assert m is not m2
    assert m == p
    assert m2 == p2
    m2 = m.transform_qubits(lambda q: q + 1, inplace=False)
    assert m is not m2
    assert m == p
    assert m2 == p2
    m2 = m.transform_qubits(lambda q: q + 1, inplace=True)
    assert m is m2
    assert m == p2
    assert m2 == p2