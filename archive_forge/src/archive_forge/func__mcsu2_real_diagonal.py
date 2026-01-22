from math import pi
from typing import Optional, Union, Tuple, List
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library.standard_gates.x import MCXGate
from qiskit.circuit.library.standard_gates.u3 import _generate_gray_code
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.exceptions import QiskitError
def _mcsu2_real_diagonal(unitary: np.ndarray, num_controls: int, ctrl_state: Optional[str]=None, use_basis_gates: bool=False) -> QuantumCircuit:
    """
    Return a multi-controlled SU(2) gate [1]_ with a real main diagonal or secondary diagonal.

    Args:
        unitary: SU(2) unitary matrix with one real diagonal.
        num_controls: The number of control qubits.
        ctrl_state: The state on which the SU(2) operation is controlled. Defaults to all
            control qubits being in state 1.
        use_basis_gates: If ``True``, use ``[p, u, cx]`` gates to implement the decomposition.

    Returns:
        A :class:`.QuantumCircuit` implementing the multi-controlled SU(2) gate.

    Raises:
        QiskitError: If the input matrix is invalid.

    References:

        .. [1]: R. Vale et al. Decomposition of Multi-controlled Special Unitary Single-Qubit Gates
            `arXiv:2302.06377 (2023) <https://arxiv.org/abs/2302.06377>`__

    """
    from .x import MCXVChain
    from qiskit.circuit.library.generalized_gates import UnitaryGate
    from qiskit.quantum_info.operators.predicates import is_unitary_matrix
    from qiskit.compiler import transpile
    if unitary.shape != (2, 2):
        raise QiskitError(f'The unitary must be a 2x2 matrix, but has shape {unitary.shape}.')
    if not is_unitary_matrix(unitary):
        raise QiskitError(f'The unitary in must be an unitary matrix, but is {unitary}.')
    is_main_diag_real = np.isclose(unitary[0, 0].imag, 0.0) and np.isclose(unitary[1, 1].imag, 0.0)
    is_secondary_diag_real = np.isclose(unitary[0, 1].imag, 0.0) and np.isclose(unitary[1, 0].imag, 0.0)
    if not is_main_diag_real and (not is_secondary_diag_real):
        raise QiskitError('The unitary must have one real diagonal.')
    if is_secondary_diag_real:
        x = unitary[0, 1]
        z = unitary[1, 1]
    else:
        x = -unitary[0, 1].real
        z = unitary[1, 1] - unitary[0, 1].imag * 1j
    if np.isclose(z, -1):
        s_op = [[1.0, 0.0], [0.0, 1j]]
    else:
        alpha_r = np.sqrt((np.sqrt((z.real + 1.0) / 2.0) + 1.0) / 2.0)
        alpha_i = z.imag / (2.0 * np.sqrt((z.real + 1.0) * (np.sqrt((z.real + 1.0) / 2.0) + 1.0)))
        alpha = alpha_r + 1j * alpha_i
        beta = x / (2.0 * np.sqrt((z.real + 1.0) * (np.sqrt((z.real + 1.0) / 2.0) + 1.0)))
        s_op = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]])
    s_gate = UnitaryGate(s_op)
    k_1 = int(np.ceil(num_controls / 2.0))
    k_2 = int(np.floor(num_controls / 2.0))
    ctrl_state_k_1 = None
    ctrl_state_k_2 = None
    if ctrl_state is not None:
        str_ctrl_state = f'{ctrl_state:0{num_controls}b}'
        ctrl_state_k_1 = str_ctrl_state[::-1][:k_1][::-1]
        ctrl_state_k_2 = str_ctrl_state[::-1][k_1:][::-1]
    circuit = QuantumCircuit(num_controls + 1, name='MCSU2')
    controls = list(range(num_controls))
    target = num_controls
    if not is_secondary_diag_real:
        circuit.h(target)
    mcx_1 = MCXVChain(num_ctrl_qubits=k_1, dirty_ancillas=True, ctrl_state=ctrl_state_k_1)
    circuit.append(mcx_1, controls[:k_1] + [target] + controls[k_1:2 * k_1 - 2])
    circuit.append(s_gate, [target])
    mcx_2 = MCXVChain(num_ctrl_qubits=k_2, dirty_ancillas=True, ctrl_state=ctrl_state_k_2)
    circuit.append(mcx_2.inverse(), controls[k_1:] + [target] + controls[k_1 - k_2 + 2:k_1])
    circuit.append(s_gate.inverse(), [target])
    mcx_3 = MCXVChain(num_ctrl_qubits=k_1, dirty_ancillas=True, ctrl_state=ctrl_state_k_1)
    circuit.append(mcx_3, controls[:k_1] + [target] + controls[k_1:2 * k_1 - 2])
    circuit.append(s_gate, [target])
    mcx_4 = MCXVChain(num_ctrl_qubits=k_2, dirty_ancillas=True, ctrl_state=ctrl_state_k_2)
    circuit.append(mcx_4, controls[k_1:] + [target] + controls[k_1 - k_2 + 2:k_1])
    circuit.append(s_gate.inverse(), [target])
    if not is_secondary_diag_real:
        circuit.h(target)
    if use_basis_gates:
        circuit = transpile(circuit, basis_gates=['p', 'u', 'cx'])
    return circuit