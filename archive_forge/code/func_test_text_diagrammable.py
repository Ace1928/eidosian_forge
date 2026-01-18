import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_text_diagrammable():
    q = cirq.NamedQubit('q')
    op0 = cirq.GateOperation(cirq.testing.SingleQubitGate(), [q])
    with pytest.raises(TypeError):
        _ = cirq.circuit_diagram_info(op0)
    op1 = cirq.GateOperation(cirq.S, [q])
    actual = cirq.circuit_diagram_info(op1)
    expected = cirq.circuit_diagram_info(cirq.S)
    assert actual == expected