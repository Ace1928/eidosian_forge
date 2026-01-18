from sympy.physics.quantum.qasm import Qasm, flip_index, trim,\
from sympy.physics.quantum.gate import X, Z, H, S, T
from sympy.physics.quantum.gate import CNOT, SWAP, CPHASE, CGate, CGateS
from sympy.physics.quantum.circuitplot import Mz
def test_qasm_readqasm():
    qasm_lines = '    qubit q_0\n    qubit q_1\n    h q_0\n    cnot q_0,q_1\n    '
    q = read_qasm(qasm_lines)
    assert q.get_circuit() == CNOT(1, 0) * H(1)