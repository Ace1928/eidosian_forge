import pennylane as qml
from pennylane.operation import Operation, AnyWires
def u2_ex_gate(phi, wires=None):
    """Implements the two-qubit exchange gate :math:`U_{2,\\mathrm{ex}}` proposed in
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ to build particle-conserving VQE ansatze
    for Quantum Chemistry simulations.

    The unitary matrix :math:`U_{2, \\mathrm{ex}}` acts on the Hilbert space of two qubits

    .. math::

        U_{2, \\mathrm{ex}}(\\phi) = \\left(\\begin{array}{cccc}
        1 & 0 & 0 & 0 \\\\
        0 & \\mathrm{cos}(\\phi) & -i\\;\\mathrm{sin}(\\phi) & 0 \\\\
        0 & -i\\;\\mathrm{sin}(\\phi) & \\mathrm{cos}(\\phi) & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        \\end{array}\\right).

    Args:
        phi (float): angle entering the controlled-RX operator :math:`CRX(2\\phi)`
        wires (list[Wires]): the two wires ``n`` and ``m`` the circuit acts on

    Returns:
        list[.Operator]: sequence of operators defined by this function
    """
    return [qml.CNOT(wires=wires), qml.CRX(2 * phi, wires=wires[::-1]), qml.CNOT(wires=wires)]