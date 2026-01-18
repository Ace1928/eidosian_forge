from math import prod
from sympy.physics.quantum.gate import H, CNOT, X, Z, CGate, CGateS, SWAP, S, T,CPHASE
from sympy.physics.quantum.circuitplot import Mz
Class to form objects from Qasm lines

    >>> from sympy.physics.quantum.qasm import Qasm
    >>> q = Qasm('qubit q0', 'qubit q1', 'h q0', 'cnot q0,q1')
    >>> q.get_circuit()
    CNOT(1,0)*H(1)
    >>> q = Qasm('qubit q0', 'qubit q1', 'cnot q0,q1', 'cnot q1,q0', 'cnot q0,q1')
    >>> q.get_circuit()
    CNOT(1,0)*CNOT(0,1)*CNOT(1,0)
    