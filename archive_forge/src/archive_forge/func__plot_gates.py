from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def _plot_gates(self):
    """Iterate through the gates and plot each of them."""
    for i, gate in enumerate(self._gates()):
        gate.plot_gate(self, i)