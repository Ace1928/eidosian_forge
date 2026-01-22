import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
def _apply_uniform_rotation_dagger(gate, alpha, control_wires, target_wire):
    """Applies a uniformly-controlled rotation to the target qubit.

    A uniformly-controlled rotation is a sequence of multi-controlled
    rotations, each of which is conditioned on the control qubits being in a different state.
    For example, a uniformly-controlled rotation with two control qubits describes a sequence of
    four multi-controlled rotations, each applying the rotation only if the control qubits
    are in states :math:`|00\\rangle`, :math:`|01\\rangle`, :math:`|10\\rangle`, and :math:`|11\\rangle`, respectively.

    To implement a uniformly-controlled rotation using single qubit rotations and CNOT gates,
    a decomposition based on Gray codes is used. For this purpose, the multi-controlled rotation
    angles alpha have to be converted into a set of non-controlled rotation angles theta.

    For more details, see `Möttönen and Vartiainen (2005), Fig 7a<https://arxiv.org/abs/quant-ph/0504100>`_.

    Args:
        gate (.Operation): gate to be applied, needs to have exactly one parameter
        alpha (tensor_like): angles to decompose the uniformly-controlled rotation into multi-controlled rotations
        control_wires (array[int]): wires that act as control
        target_wire (int): wire that acts as target

    Returns:
          list[.Operator]: sequence of operators defined by this function
    """
    op_list = []
    theta = compute_theta(alpha)
    gray_code_rank = len(control_wires)
    if gray_code_rank == 0:
        if qml.math.is_abstract(theta) or qml.math.all(theta[..., 0] != 0.0):
            op_list.append(gate(theta[..., 0], wires=[target_wire]))
        return op_list
    code = gray_code(gray_code_rank)
    num_selections = len(code)
    control_indices = [int(np.log2(int(code[i], 2) ^ int(code[(i + 1) % num_selections], 2))) for i in range(num_selections)]
    for i, control_index in enumerate(control_indices):
        if qml.math.is_abstract(theta) or qml.math.all(theta[..., i] != 0.0):
            op_list.append(gate(theta[..., i], wires=[target_wire]))
        op_list.append(qml.CNOT(wires=[control_wires[control_index], target_wire]))
    return op_list