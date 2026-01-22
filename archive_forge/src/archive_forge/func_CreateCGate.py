from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def CreateCGate(name, latexname=None):
    """Use a lexical closure to make a controlled gate.
    """
    if not latexname:
        latexname = name
    onequbitgate = CreateOneQubitGate(name, latexname)

    def ControlledGate(ctrls, target):
        return CGate(tuple(ctrls), onequbitgate(target))
    return ControlledGate