import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_with_gate():
    g1 = cirq.GateOperation(cirq.X, cirq.LineQubit.range(1))
    g2 = cirq.GateOperation(cirq.Y, cirq.LineQubit.range(1))
    assert g1.with_gate(cirq.X) is g1
    assert g1.with_gate(cirq.Y) == g2