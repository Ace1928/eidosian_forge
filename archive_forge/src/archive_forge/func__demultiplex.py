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
def _demultiplex(um0, um1, opt_a1=False, opt_a2=False, *, _depth=0):
    """Decompose a generic multiplexer.

          ────□────
           ┌──┴──┐
         /─┤     ├─
           └─────┘

    represented by the block diagonal matrix

            ┏         ┓
            ┃ um0     ┃
            ┃     um1 ┃
            ┗         ┛

    to
               ┌───┐
        ───────┤ Rz├──────
          ┌───┐└─┬─┘┌───┐
        /─┤ w ├──□──┤ v ├─
          └───┘     └───┘

    where v and w are general unitaries determined from decomposition.

    Args:
       um0 (ndarray): applied if MSB is 0
       um1 (ndarray): applied if MSB is 1
       opt_a1 (bool): whether to try optimization A.1 from Shende. This should eliminate 1 cnot
          per call. If True CZ gates are left in the output. If desired these can be further decomposed
       opt_a2 (bool): whether to try  optimization A.2 from Shende. This decomposes two qubit
          unitaries into a diagonal gate and a two cx unitary and reduces overall cx count by
          4^(n-2) - 1.
       _depth (int): This is an internal variable to track the recursion depth.

    Returns:
        QuantumCircuit: decomposed circuit
    """
    dim = um0.shape[0] + um1.shape[0]
    nqubits = int(np.log2(dim))
    um0um1 = um0 @ um1.T.conjugate()
    if is_hermitian_matrix(um0um1):
        eigvals, vmat = scipy.linalg.eigh(um0um1)
    else:
        evals, vmat = scipy.linalg.schur(um0um1, output='complex')
        eigvals = evals.diagonal()
    dvals = np.emath.sqrt(eigvals)
    dmat = np.diag(dvals)
    wmat = dmat @ vmat.T.conjugate() @ um1
    circ = QuantumCircuit(nqubits)
    left_gate = qs_decomposition(wmat, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth + 1).to_instruction()
    circ.append(left_gate, range(nqubits - 1))
    angles = 2 * np.angle(np.conj(dvals))
    ucrz = UCRZGate(angles.tolist())
    circ.append(ucrz, [nqubits - 1] + list(range(nqubits - 1)))
    right_gate = qs_decomposition(vmat, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth + 1).to_instruction()
    circ.append(right_gate, range(nqubits - 1))
    return circ