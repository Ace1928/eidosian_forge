import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_works_on_operation():

    class XAsOp(cirq.Operation):

        def __init__(self, q):
            self.q = q

        @property
        def qubits(self):
            return (self.q,)

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

        def _kraus_(self):
            return cirq.kraus(cirq.X)
    s = cirq.DensityMatrixSimulator()
    c = cirq.Circuit(XAsOp(cirq.LineQubit(0)))
    np.testing.assert_allclose(s.simulate(c).final_density_matrix, np.diag([0, 1]), atol=1e-08)