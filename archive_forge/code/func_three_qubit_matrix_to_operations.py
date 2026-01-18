from typing import Union, Tuple, Sequence, List, Optional
import numpy as np
import cirq
from cirq import ops
from cirq import transformers as opt
def three_qubit_matrix_to_operations(q0: ops.Qid, q1: ops.Qid, q2: ops.Qid, u: np.ndarray, atol: float=1e-08) -> Sequence[ops.Operation]:
    """Returns operations for a 3 qubit unitary.

    The algorithm is described in Shende et al.:
    Synthesis of Quantum Logic Circuits. Tech. rep. 2006,
    https://arxiv.org/abs/quant-ph/0406176

    Args:
        q0: first qubit
        q1: second qubit
        q2: third qubit
        u: unitary matrix
        atol: A limit on the amount of absolute error introduced by the
            construction.

    Returns:
        The resulting operations will have only known two-qubit and one-qubit
        gates based operations, namely CZ, CNOT and rx, ry, PhasedXPow gates.

    Raises:
        ValueError: If the u matrix is non-unitary or not of shape (8,8).
        ImportError: If the decomposition cannot be done because the SciPy version is less than
            1.5.0 and so does not contain the required `cossin` method.
    """
    if np.shape(u) != (8, 8):
        raise ValueError(f'Expected unitary matrix with shape (8,8) got {np.shape(u)}')
    if not cirq.is_unitary(u, atol=atol):
        raise ValueError(f'Matrix is not unitary: {u}')
    try:
        from scipy.linalg import cossin
    except ImportError:
        raise ImportError('cirq.three_qubit_unitary_to_operations requires SciPy 1.5.0+, as it uses the cossin function. Please upgrade scipy in your environment to use this function!')
    (u1, u2), theta, (v1h, v2h) = cossin(u, 4, 4, separate=True)
    cs_ops = _cs_to_ops(q0, q1, q2, theta)
    if len(cs_ops) > 0 and cs_ops[-1] == cirq.CZ(q2, q0):
        u2 = u2 @ np.diag([1, -1, 1, -1])
        cs_ops = cs_ops[:-1]
    d_ud, ud_ops = _two_qubit_multiplexor_to_ops(q0, q1, q2, u1, u2, shift_left=True, atol=atol)
    _, vdh_ops = _two_qubit_multiplexor_to_ops(q0, q1, q2, v1h, v2h, shift_left=False, diagonal=d_ud, atol=atol)
    return list(cirq.Circuit(vdh_ops + cs_ops + ud_ops).all_operations())