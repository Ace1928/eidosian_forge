from sympy.physics.quantum.qasm import Qasm, flip_index, trim,\
from sympy.physics.quantum.gate import X, Z, H, S, T
from sympy.physics.quantum.gate import CNOT, SWAP, CPHASE, CGate, CGateS
from sympy.physics.quantum.circuitplot import Mz
def test_qasm_stripquotes():
    assert stripquotes("'S'") == 'S'
    assert stripquotes('"S"') == 'S'
    assert stripquotes('S') == 'S'