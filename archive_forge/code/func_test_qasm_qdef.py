from sympy.physics.quantum.qasm import Qasm, flip_index, trim,\
from sympy.physics.quantum.gate import X, Z, H, S, T
from sympy.physics.quantum.gate import CNOT, SWAP, CPHASE, CGate, CGateS
from sympy.physics.quantum.circuitplot import Mz
def test_qasm_qdef():
    q = Qasm('def Q,0,Q', 'qubit q0', 'Q q0')
    assert str(q.get_circuit()) == 'Q(0)'
    q = Qasm('def CQ,1,Q', 'qubit q0', 'qubit q1', 'CQ q0,q1')
    assert str(q.get_circuit()) == 'C((1),Q(0))'