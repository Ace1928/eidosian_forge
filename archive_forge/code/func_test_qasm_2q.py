from sympy.physics.quantum.qasm import Qasm, flip_index, trim,\
from sympy.physics.quantum.gate import X, Z, H, S, T
from sympy.physics.quantum.gate import CNOT, SWAP, CPHASE, CGate, CGateS
from sympy.physics.quantum.circuitplot import Mz
def test_qasm_2q():
    for symbol, gate in [('cnot', CNOT), ('swap', SWAP), ('cphase', CPHASE)]:
        q = Qasm('qubit q_0', 'qubit q_1', '%s q_0,q_1' % symbol)
        assert q.get_circuit() == gate(1, 0)