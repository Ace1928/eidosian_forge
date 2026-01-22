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
class IsingZZ(Operation):
    """
    Ising ZZ coupling gate

    .. math:: ZZ(\\phi) = \\exp(-i \\frac{\\phi}{2} (Z \\otimes Z)) =
        \\begin{bmatrix}
            e^{-i \\phi / 2} & 0 & 0 & 0 \\\\
            0 & e^{i \\phi / 2} & 0 & 0 \\\\
            0 & 0 & e^{i \\phi / 2} & 0 \\\\
            0 & 0 & 0 & e^{-i \\phi / 2}
        \\end{bmatrix}.

    .. note::

        Special cases of using the :math:`ZZ` operator include:

        * :math:`ZZ(0) = I`;
        * :math:`ZZ(\\pi) = - (Z \\otimes Z)`;
        * :math:`ZZ(2\\pi) = - I`;

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\\frac{d}{d\\phi}f(ZZ(\\phi)) = \\frac{1}{2}\\left[f(ZZ(\\phi +\\pi/2)) - f(ZZ(\\phi-\\pi/2))\\right]`
      where :math:`f` is an expectation value depending on :math:`ZZ(\\theta)`.

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
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
        return -0.5 * PauliZ(wires=self.wires[0]) @ PauliZ(wires=self.wires[1])

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.IsingZZ.decomposition`.

        Args:
            phi (float): the phase angle
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.IsingZZ.compute_decomposition(1.23, wires=[0, 1])
        [CNOT(wires=[0, 1]), RZ(1.23, wires=[1]), CNOT(wires=[0, 1])]

        """
        return [qml.CNOT(wires=wires), RZ(phi, wires=[wires[1]]), qml.CNOT(wires=wires)]

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.IsingZZ.matrix`


        Args:
           phi (tensor_like or float): phase angle

        Returns:
           tensor_like: canonical matrix

        **Example**

        >>> qml.IsingZZ.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689-0.2474j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.9689+0.2474j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9689-0.2474j]])
        """
        if qml.math.get_interface(phi) == 'tensorflow':
            p = qml.math.exp(-0.5j * qml.math.cast_like(phi, 1j))
            if qml.math.ndim(p) == 0:
                return qml.math.diag([p, qml.math.conj(p), qml.math.conj(p), p])
            diags = stack_last([p, qml.math.conj(p), qml.math.conj(p), p])
            return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)
        signs = qml.math.array([1, -1, -1, 1], like=phi)
        arg = -0.5j * phi
        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * signs))
        diags = qml.math.exp(qml.math.outer(arg, signs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)

    @staticmethod
    def compute_eigvals(phi):
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.IsingZZ.eigvals`


        Args:
            phi (tensor_like or float): phase angle

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.IsingZZ.compute_eigvals(torch.tensor(0.5))
        tensor([0.9689-0.2474j, 0.9689+0.2474j, 0.9689+0.2474j, 0.9689-0.2474j])
        """
        if qml.math.get_interface(phi) == 'tensorflow':
            phase = qml.math.exp(-0.5j * qml.math.cast_like(phi, 1j))
            return stack_last([phase, qml.math.conj(phase), qml.math.conj(phase), phase])
        prefactors = qml.math.array([-0.5j, 0.5j, 0.5j, -0.5j], like=phi)
        if qml.math.ndim(phi) == 0:
            product = phi * prefactors
        else:
            product = qml.math.outer(phi, prefactors)
        return qml.math.exp(product)

    def adjoint(self):
        phi, = self.parameters
        return IsingZZ(-phi, wires=self.wires)

    def pow(self, z):
        return [IsingZZ(self.data[0] * z, wires=self.wires)]

    def simplify(self):
        phi = self.data[0] % (4 * np.pi)
        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])
        return IsingZZ(phi, wires=self.wires)