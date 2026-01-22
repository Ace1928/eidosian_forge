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
class MultiRZ(Operation):
    """
    Arbitrary multi Z rotation.

    .. math::

        MultiRZ(\\theta) = \\exp(-i \\frac{\\theta}{2} Z^{\\otimes n})

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\\frac{d}{d\\theta}f(MultiRZ(\\theta)) = \\frac{1}{2}\\left[f(MultiRZ(\\theta +\\pi/2)) - f(MultiRZ(\\theta-\\pi/2))\\right]`
      where :math:`f` is an expectation value depending on :math:`MultiRZ(\\theta)`.

    .. note::

        If the ``MultiRZ`` gate is not supported on the targeted device, PennyLane
        will decompose the gate using :class:`~.RZ` and :class:`~.CNOT` gates.

    Args:
        theta (tensor_like or float): rotation angle :math:`\\theta`
        wires (Sequence[int] or int): the wires the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_wires = AnyWires
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = 'A'
    parameter_frequencies = [(1,)]

    def _flatten(self):
        return (self.data, (self.wires, tuple()))

    def __init__(self, theta, wires=None, id=None):
        wires = Wires(wires)
        self.hyperparameters['num_wires'] = len(wires)
        super().__init__(theta, wires=wires, id=id)

    @staticmethod
    def compute_matrix(theta, num_wires):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.MultiRZ.matrix`

        Args:
            theta (tensor_like or float): rotation angle
            num_wires (int): number of wires the rotation acts on

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.MultiRZ.compute_matrix(torch.tensor(0.1), 2)
        tensor([[0.9988-0.0500j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.9988+0.0500j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.9988+0.0500j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9988-0.0500j]])
        """
        eigs = qml.math.convert_like(pauli_eigs(num_wires), theta)
        if qml.math.get_interface(theta) == 'tensorflow':
            theta = qml.math.cast_like(theta, 1j)
            eigs = qml.math.cast_like(eigs, 1j)
        if qml.math.ndim(theta) == 0:
            return qml.math.diag(qml.math.exp(-0.5j * theta * eigs))
        diags = qml.math.exp(qml.math.outer(-0.5j * theta, eigs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(2 ** num_wires, like=diags), diags)

    def generator(self):
        return -0.5 * functools.reduce(matmul, [PauliZ(w) for w in self.wires])

    @staticmethod
    def compute_eigvals(theta, num_wires):
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.MultiRZ.eigvals`


        Args:
            theta (tensor_like or float): rotation angle
            num_wires (int): number of wires the rotation acts on

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.MultiRZ.compute_eigvals(torch.tensor(0.5), 3)
        tensor([0.9689-0.2474j, 0.9689+0.2474j, 0.9689+0.2474j, 0.9689-0.2474j,
                0.9689+0.2474j, 0.9689-0.2474j, 0.9689-0.2474j, 0.9689+0.2474j])
        """
        eigs = qml.math.convert_like(pauli_eigs(num_wires), theta)
        if qml.math.get_interface(theta) == 'tensorflow':
            theta = qml.math.cast_like(theta, 1j)
            eigs = qml.math.cast_like(eigs, 1j)
        if qml.math.ndim(theta) == 0:
            return qml.math.exp(-0.5j * theta * eigs)
        return qml.math.exp(qml.math.outer(-0.5j * theta, eigs))

    @staticmethod
    def compute_decomposition(theta, wires, **kwargs):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.MultiRZ.decomposition`.

        Args:
            theta (float): rotation angle :math:`\\theta`
            wires (Iterable, Wires): the wires the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.MultiRZ.compute_decomposition(1.2, wires=(0,1))
        [CNOT(wires=[1, 0]), RZ(1.2, wires=[0]), CNOT(wires=[1, 0])]

        """
        ops = [qml.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[~0:0:-1], wires[~1::-1])]
        ops.append(RZ(theta, wires=wires[0]))
        ops += [qml.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[1:], wires[:~0])]
        return ops

    def adjoint(self):
        return MultiRZ(-self.parameters[0], wires=self.wires)

    def pow(self, z):
        return [MultiRZ(self.data[0] * z, wires=self.wires)]

    def simplify(self):
        theta = self.data[0] % (4 * np.pi)
        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires[0])
        return MultiRZ(theta, wires=self.wires)