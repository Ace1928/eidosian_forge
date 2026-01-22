import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import RZ, RX, CNOT, Hadamard
def _layer8(weight, s, r, q, p, set_cnot_wires):
    """Implement the eighth layer of the circuit to exponentiate the double-excitation
    operator entering the UCCSD ansatz.

    .. math::

        \\hat{U}_{pqrs}^{(8)}(\\theta) = \\mathrm{exp} \\Big\\{ -\\frac{i\\theta}{8}
        \\bigotimes_{b=s+1}^{r-1} \\hat{Z}_b \\bigotimes_{a=q+1}^{p-1} \\hat{Z}_a
        (\\hat{Y}_s \\hat{Y}_r \\hat{X}_q \\hat{Y}_p) \\Big\\}

    Args:
        weight (float): angle :math:`\\theta` entering the Z rotation acting on wire ``p``
        s (int): qubit index ``s``
        r (int): qubit index ``r``
        q (int): qubit index ``q``
        p (int): qubit index ``p``
        set_cnot_wires (list[Wires]): list of CNOT wires

    Returns:
        list[.Operator]: sequence of operators defined by this function
    """
    op_list = [RX(-np.pi / 2, wires=s), RX(-np.pi / 2, wires=r), Hadamard(wires=q), RX(-np.pi / 2, wires=p)]
    for cnot_wires in set_cnot_wires:
        op_list.append(CNOT(wires=cnot_wires))
    op_list.append(RZ(-weight / 8, wires=p))
    for cnot_wires in reversed(set_cnot_wires):
        op_list.append(CNOT(wires=cnot_wires))
    op_list.append(RX(np.pi / 2, wires=s))
    op_list.append(RX(np.pi / 2, wires=r))
    op_list.append(Hadamard(wires=q))
    op_list.append(RX(np.pi / 2, wires=p))
    return op_list