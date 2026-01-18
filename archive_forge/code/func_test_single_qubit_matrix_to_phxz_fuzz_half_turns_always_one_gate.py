import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('pre_turns,post_turns', [(random.random(), random.random()) for _ in range(10)])
def test_single_qubit_matrix_to_phxz_fuzz_half_turns_always_one_gate(pre_turns, post_turns):
    atol = 1e-06
    aggr_atol = atol * 10.0
    intended_effect = cirq.dot(cirq.unitary(cirq.Z ** (2 * pre_turns)), cirq.unitary(cirq.X), cirq.unitary(cirq.Z ** (2 * post_turns)))
    gate = cirq.single_qubit_matrix_to_phxz(intended_effect, atol=atol)
    assert gate.z_exponent == 0
    assert_gates_implement_unitary([gate], intended_effect, atol=aggr_atol)