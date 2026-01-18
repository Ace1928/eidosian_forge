from typing import Optional, TYPE_CHECKING, Set, List
import pytest
import cirq
from cirq import PointOptimizer, PointOptimizationSummary, Operation
from cirq.testing import EqualsTester
def test_point_optimizer_can_write_new_gates_inline():
    x = cirq.NamedQubit('x')
    y = cirq.NamedQubit('y')
    z = cirq.NamedQubit('z')
    c = cirq.Circuit(cirq.CZ(x, y), cirq.Y(x), cirq.Z(x), cirq.X(y), cirq.CNOT(y, z), cirq.Z(y), cirq.Z(x), cirq.CNOT(y, z), cirq.CNOT(z, y))
    ReplaceWithXGates()(c)
    actual_text_diagram = c.to_text_diagram().strip()
    expected_text_diagram = '\nx: ───X───X───X───X───────────\n\ny: ───X───X───────X───X───X───\n\nz: ───────────────────X───X───\n    '.strip()
    assert actual_text_diagram == expected_text_diagram