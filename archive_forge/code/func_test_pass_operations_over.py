import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_pass_operations_over():
    q0, q1 = _make_qubits(2)
    op = cirq.SingleQubitCliffordGate.from_double_map({cirq.Z: (cirq.X, False), cirq.X: (cirq.Z, False)})(q0)
    ps_before = cirq.PauliString({q0: cirq.X, q1: cirq.Y}, -1)
    ps_after = cirq.PauliString({q0: cirq.Z, q1: cirq.Y}, -1)
    before = cirq.PauliStringPhasor(ps_before, exponent_neg=0.1)
    after = cirq.PauliStringPhasor(ps_after, exponent_neg=0.1)
    assert before.pass_operations_over([op]) == after
    assert after.pass_operations_over([op], after_to_before=True) == before