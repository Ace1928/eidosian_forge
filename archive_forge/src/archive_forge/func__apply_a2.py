from __future__ import annotations
from typing import Callable
import scipy
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, QuantumRegister
from qiskit.synthesis.two_qubit import (
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.circuit.library.generalized_gates.uc_pauli_rot import UCPauliRotGate, _EPS
from qiskit.circuit.library.generalized_gates.ucry import UCRYGate
from qiskit.circuit.library.generalized_gates.ucrz import UCRZGate
def _apply_a2(circ):
    from qiskit.compiler import transpile
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
    decomposer = two_qubit_decompose.TwoQubitDecomposeUpToDiagonal()
    ccirc = transpile(circ, basis_gates=['u', 'cx', 'qsd2q'], optimization_level=0)
    ind2q = []
    for i, instruction in enumerate(ccirc.data):
        if instruction.operation.name == 'qsd2q':
            ind2q.append(i)
    if len(ind2q) == 0:
        return ccirc
    elif len(ind2q) == 1:
        ccirc.data[ind2q[0]].operation.name = 'Unitary'
        return ccirc
    ind2 = None
    for ind1, ind2 in zip(ind2q[0:-1], ind2q[1:]):
        instr1 = ccirc.data[ind1]
        mat1 = Operator(instr1.operation).data
        instr2 = ccirc.data[ind2]
        mat2 = Operator(instr2.operation).data
        dmat, qc2cx = decomposer(mat1)
        ccirc.data[ind1] = instr1.replace(operation=qc2cx.to_gate())
        mat2 = mat2 @ dmat
        ccirc.data[ind2] = instr2.replace(UnitaryGate(mat2))
    qc3 = two_qubit_decompose.two_qubit_cnot_decompose(mat2)
    ccirc.data[ind2] = ccirc.data[ind2].replace(operation=qc3.to_gate())
    return ccirc