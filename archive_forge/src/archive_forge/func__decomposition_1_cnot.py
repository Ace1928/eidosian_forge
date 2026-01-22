import numpy as np
import pennylane as qml
from pennylane import math
from .single_qubit_unitary import one_qubit_decomposition
def _decomposition_1_cnot(U, wires):
    """If there is just one CNOT, we can write the circuit in the form
     -╭U- = -C--╭C--A-
     -╰U- = -D--╰X--B-

    To do this decomposition, first we find G, H in SO(4) such that
        G (Edag V E) H = (Edag U E)

    where V depends on the central CNOT gate, and both U, V are in SU(4). This
    is done following the methods in https://arxiv.org/abs/quant-ph/0308045.

    Once we find G and H, we can use the fact that E SO(4) Edag gives us
    something in SU(2) x SU(2) to give A, B, C, D.
    """
    swap_U = np.exp(1j * np.pi / 4) * math.dot(math.cast_like(SWAP, U), U)
    u = math.dot(math.cast_like(Edag, U), math.dot(swap_U, math.cast_like(E, U)))
    uuT = math.dot(u, math.T(u))
    _, p = math.linalg.eigh(qml.math.real(uuT))
    p = math.dot(p, math.diag([1, 1, 1, math.sign(math.linalg.det(p))]))
    G = math.dot(p, q_one_cnot.T)
    H = math.dot(math.conj(math.T(v_one_cnot)), math.dot(math.T(G), u))
    AB = math.dot(E, math.dot(G, Edag))
    CD = math.dot(E, math.dot(H, Edag))
    A, B = _su2su2_to_tensor_products(AB)
    C, D = _su2su2_to_tensor_products(CD)
    A_ops = one_qubit_decomposition(A, wires[1])
    B_ops = one_qubit_decomposition(B, wires[0])
    C_ops = one_qubit_decomposition(C, wires[0])
    D_ops = one_qubit_decomposition(D, wires[1])
    return C_ops + D_ops + [qml.CNOT(wires=wires)] + A_ops + B_ops