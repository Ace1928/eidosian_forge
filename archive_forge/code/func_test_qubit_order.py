import itertools
import numpy as np
import pytest
import cirq
import sympy
@pytest.mark.parametrize('u', TWO_SQRT_ISWAP_UNITARIES[:1])
def test_qubit_order(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqrt_iswap_operations(q1, q0, u, required_sqrt_iswap_count=2)
    assert_valid_decomp(u, ops, qubit_order=(q1, q0))
    assert_specific_sqrt_iswap_count(ops, 2)