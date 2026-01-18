from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def test_not_decompose_czs():
    circuit = cirq.Circuit(cirq.CZPowGate(exponent=1, global_shift=-0.5).on(*cirq.LineQubit.range(2)))
    assert_optimizes(before=circuit, expected=circuit)