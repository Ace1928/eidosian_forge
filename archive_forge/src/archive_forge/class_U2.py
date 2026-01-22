import functools
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
class U2(Operation):
    """
    U2 gate.

    .. math::

        U_2(\\phi, \\delta) = \\frac{1}{\\sqrt{2}}\\begin{bmatrix} 1 & -\\exp(i \\delta)
        \\\\ \\exp(i \\phi) & \\exp(i (\\phi + \\delta)) \\end{bmatrix}

    The :math:`U_2` gate is related to the single-qubit rotation :math:`R` (:class:`Rot`) and the
    :math:`R_\\phi` (:class:`PhaseShift`) gates via the following relation:

    .. math::

        U_2(\\phi, \\delta) = R_\\phi(\\phi+\\delta) R(\\delta,\\pi/2,-\\delta)

    .. note::

        If the ``U2`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.Rot` and :class:`~.PhaseShift` gates.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Number of dimensions per parameter: (0, 0)
    * Gradient recipe: :math:`\\frac{d}{d\\phi}f(U_2(\\phi, \\delta)) = \\frac{1}{2}\\left[f(U_2(\\phi+\\pi/2, \\delta)) - f(U_2(\\phi-\\pi/2, \\delta))\\right]`
      where :math:`f` is an expectation value depending on :math:`U_2(\\phi, \\delta)`.
      This gradient recipe applies for each angle argument :math:`\\{\\phi, \\delta\\}`.

    Args:
        phi (float): azimuthal angle :math:`\\phi`
        delta (float): quantum phase :math:`\\delta`
        wires (Sequence[int] or int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """
    num_wires = 1
    num_params = 2
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0, 0)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = 'A'
    parameter_frequencies = [(1,), (1,)]

    def __init__(self, phi, delta, wires, id=None):
        super().__init__(phi, delta, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi, delta):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.U2.matrix`

        Args:
            phi (tensor_like or float): azimuthal angle
            delta (tensor_like or float): quantum phase

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.U2.compute_matrix(torch.tensor(0.1), torch.tensor(0.2))
        tensor([[ 0.7071+0.0000j, -0.6930-0.1405j],
                [ 0.7036+0.0706j,  0.6755+0.2090j]])
        """
        interface = qml.math.get_interface(phi, delta)
        if interface == 'tensorflow':
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            delta = qml.math.cast_like(qml.math.asarray(delta, like=interface), 1j)
        one = qml.math.ones_like(phi) * qml.math.ones_like(delta)
        mat = [[one, -qml.math.exp(1j * delta) * one], [qml.math.exp(1j * phi) * one, qml.math.exp(1j * (phi + delta))]]
        return qml.math.stack([stack_last(row) for row in mat], axis=-2) / np.sqrt(2)

    @staticmethod
    def compute_decomposition(phi, delta, wires):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.

        .. seealso:: :meth:`~.U2.decomposition`.

        Args:
            phi (float): azimuthal angle :math:`\\phi`
            delta (float): quantum phase :math:`\\delta`
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.U2.compute_decomposition(1.23, 2.34, wires=0)
        [Rot(2.34, 1.5707963267948966, -2.34, wires=[0]),
        PhaseShift(2.34, wires=[0]),
        PhaseShift(1.23, wires=[0])]

        """
        pi_half = qml.math.ones_like(delta) * (np.pi / 2)
        return [Rot(delta, pi_half, -delta, wires=wires), PhaseShift(delta, wires=wires), PhaseShift(phi, wires=wires)]

    def adjoint(self):
        phi, delta = self.parameters
        new_delta = qml.math.mod(np.pi - phi, 2 * np.pi)
        new_phi = qml.math.mod(np.pi - delta, 2 * np.pi)
        return U2(new_phi, new_delta, wires=self.wires)

    def simplify(self):
        """Simplifies the gate into RX or RY gates if possible."""
        wires = self.wires
        phi, delta = [p % (2 * np.pi) for p in self.data]
        if _can_replace(delta, 0) and _can_replace(phi, 0):
            return RY(np.pi / 2, wires=wires)
        if _can_replace(delta, np.pi / 2) and _can_replace(phi, 3 * np.pi / 2):
            return RX(np.pi / 2, wires=wires)
        if _can_replace(delta, 3 * np.pi / 2) and _can_replace(phi, np.pi / 2):
            return RX(3 * np.pi / 2, wires=wires)
        return U2(phi, delta, wires=wires)