import functools
from operator import matmul
import numpy as np
import pennylane as qml
from pennylane.math import expand_matrix
from pennylane.operation import AnyWires, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires
from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
from .parametric_ops_single_qubit import _can_replace, stack_last, RX, RY, RZ, PhaseShift
class CPhaseShift10(Operation):
    """
    A qubit controlled phase shift.

    .. math:: CR_{10\\phi}(\\phi) = \\begin{bmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 1 & 0 & 0 \\\\
                0 & 0 & e^{i\\phi} & 0 \\\\
                0 & 0 & 0 & 1
            \\end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe:

    .. math::
        \\frac{d}{d \\phi} CR_{10}(\\phi)
        = \\frac{1}{2} \\left[ CR_{10}(\\phi + \\pi / 2)
            - CR_{10}(\\phi - \\pi / 2) \\right]

    Args:
        phi (float): rotation angle :math:`\\phi`
        wires (Any, Wires): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = 'A'
    parameter_frequencies = [(1,)]

    def generator(self):
        return qml.Projector(np.array([1, 0]), wires=self.wires)

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label='Rϕ(10)', cache=cache)

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CPhaseShift10.matrix`

        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.CPhaseShift10.compute_matrix(torch.tensor(0.5))
            tensor([[1.0+0.0j, 0.0+0.0j, 0.0000+0.0000j, 0.0+0.0j],
                    [0.0+0.0j, 1.0+0.0j, 0.0000+0.0000j, 0.0+0.0j],
                    [0.0+0.0j, 0.0+0.0j, 0.8776+0.4794j, 0.0+0.0j],
                    [0.0+0.0j, 0.0+0.0j, 0.0000+0.0000j, 1.0+0.0j]])
        """
        if qml.math.get_interface(phi) == 'tensorflow':
            phi = qml.math.cast_like(phi, 1j)
        exp_part = qml.math.exp(1j * phi)
        if qml.math.ndim(phi) > 0:
            ones = qml.math.ones_like(exp_part)
            zeros = qml.math.zeros_like(exp_part)
            matrix = [[ones, zeros, zeros, zeros], [zeros, ones, zeros, zeros], [zeros, zeros, exp_part, zeros], [zeros, zeros, zeros, ones]]
            return qml.math.stack([stack_last(row) for row in matrix], axis=-2)
        return qml.math.diag([1, 1, exp_part, 1])

    @staticmethod
    def compute_eigvals(phi):
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.CPhaseShift10.eigvals`


        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.CPhaseShift10.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 1.0000+0.0000j, 0.8776+0.4794j, 1.0000+0.0000j])
        """
        if qml.math.get_interface(phi) == 'tensorflow':
            phi = qml.math.cast_like(phi, 1j)
        exp_part = qml.math.exp(1j * phi)
        ones = qml.math.ones_like(exp_part)
        return stack_last([ones, ones, exp_part, ones])

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.
        .. seealso:: :meth:`~.CPhaseShift10.decomposition`.

        Args:
            phi (float): rotation angle :math:`\\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CPhaseShift10.compute_decomposition(1.234, wires=(0,1))
        [X(1),
        PhaseShift(0.617, wires=[0]),
        PhaseShift(0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        PhaseShift(-0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        X(1)]

        """
        decomp_ops = [PauliX(wires[1]), PhaseShift(phi / 2, wires=[wires[0]]), PhaseShift(phi / 2, wires=[wires[1]]), qml.CNOT(wires=wires), PhaseShift(-phi / 2, wires=[wires[1]]), qml.CNOT(wires=wires), PauliX(wires[1])]
        return decomp_ops

    def adjoint(self):
        return CPhaseShift10(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [CPhaseShift10(self.data[0] * z, wires=self.wires)]

    @property
    def control_wires(self):
        return self.wires[0:1]