from typing import Optional, TYPE_CHECKING, Set, List
import pytest
import cirq
from cirq import PointOptimizer, PointOptimizationSummary, Operation
from cirq.testing import EqualsTester
def test_point_optimizer_post_clean_up():
    x = cirq.NamedQubit('x')
    y = cirq.NamedQubit('y')
    z = cirq.NamedQubit('z')
    c = cirq.Circuit(cirq.CZ(x, y), cirq.Y(x), cirq.Z(x), cirq.X(y), cirq.CNOT(y, z), cirq.Z(y), cirq.Z(x), cirq.CNOT(y, z), cirq.CNOT(z, y))

    def clean_up(operations):
        for op in operations:
            yield (op ** 0.5)
    ReplaceWithXGates(post_clean_up=clean_up)(c)
    actual_text_diagram = c.to_text_diagram().strip()
    expected_text_diagram = '\nx: ───X^0.5───X^0.5───X^0.5───X^0.5───────────────────\n\ny: ───X^0.5───X^0.5───────────X^0.5───X^0.5───X^0.5───\n\nz: ───────────────────────────────────X^0.5───X^0.5───\n    '.strip()
    assert actual_text_diagram == expected_text_diagram