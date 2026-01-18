import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('expression, expected_result', ((cirq.LinearCombinationOfOperations({cirq.XX(q0, q1): 2}), cirq.LinearCombinationOfOperations({cirq.ParallelGate(cirq.X, 2).on(q0, q1): 2})), (cirq.LinearCombinationOfOperations({cirq.CNOT(q0, q1): 2}), cirq.LinearCombinationOfOperations({cirq.IdentityGate(2).on(q0, q1): 1, cirq.PauliString({q1: cirq.X}): 1, cirq.PauliString({q0: cirq.Z}): 1, cirq.PauliString({q0: cirq.Z, q1: cirq.X}): -1})), (cirq.LinearCombinationOfOperations({cirq.X(q0): 1}) ** 2, cirq.LinearCombinationOfOperations({cirq.I(q0): 1})), (cirq.LinearCombinationOfOperations({cirq.X(q0): np.sqrt(0.5), cirq.Y(q0): np.sqrt(0.5)}) ** 2, cirq.LinearCombinationOfOperations({cirq.I(q0): 1})), (cirq.LinearCombinationOfOperations({cirq.X(q0): 1, cirq.Z(q0): 1}) ** 3, cirq.LinearCombinationOfOperations({cirq.X(q0): 2, cirq.Z(q0): 2})), (cirq.LinearCombinationOfOperations({cirq.X(q0): 1j, cirq.Y(q0): 1}) ** 2, cirq.LinearCombinationOfOperations({})), (cirq.LinearCombinationOfOperations({cirq.X(q0): -1j, cirq.Y(q0): 1}) ** 2, cirq.LinearCombinationOfOperations({})), (cirq.LinearCombinationOfOperations({cirq.X(q0): 3 / 13, cirq.Y(q0): -4 / 13, cirq.Z(q0): 12 / 13}) ** 24, cirq.LinearCombinationOfOperations({cirq.I(q0): 1})), (cirq.LinearCombinationOfOperations({cirq.X(q0): 3 / 13, cirq.Y(q0): -4 / 13, cirq.Z(q0): 12 / 13}) ** 25, cirq.LinearCombinationOfOperations({cirq.X(q0): 3 / 13, cirq.Y(q0): -4 / 13, cirq.Z(q0): 12 / 13})), (cirq.LinearCombinationOfOperations({cirq.X(q1): 2, cirq.Z(q1): 3}) ** 0, cirq.LinearCombinationOfOperations({cirq.I(q1): 1}))))
def test_operation_expressions(expression, expected_result):
    assert_linear_combinations_are_equal(expression, expected_result)