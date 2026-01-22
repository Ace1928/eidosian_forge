from quantum chemistry applications.
import functools
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane.operation import Operation
class SingleExcitation(Operation):
    """
    Single excitation rotation.

    .. math:: U(\\phi) = \\begin{bmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & \\cos(\\phi/2) & -\\sin(\\phi/2) & 0 \\\\
                0 & \\sin(\\phi/2) & \\cos(\\phi/2) & 0 \\\\
                0 & 0 & 0 & 1
            \\end{bmatrix}.

    This operation performs a rotation in the two-dimensional subspace :math:`\\{|01\\rangle,
    |10\\rangle\\}`. The name originates from the occupation-number representation of
    fermionic wavefunctions, where the transformation  from :math:`|10\\rangle` to :math:`|01\\rangle`
    is interpreted as "exciting" a particle from the first qubit to the second.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The ``SingleExcitation`` operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

    Args:
        phi (float): rotation angle :math:`\\phi`
        wires (Sequence[int]): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    **Example**

    The following circuit performs the transformation :math:`|10\\rangle\\rightarrow \\cos(
    \\phi/2)|10\\rangle -\\sin(\\phi/2)|01\\rangle`:

    .. code-block::

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(phi):
            qml.X(0)
            qml.SingleExcitation(phi, wires=[0, 1])
            return qml.state()

        circuit(0.1)
    """
    num_wires = 2
    'int: Number of wires that the operator acts on.'
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = 'A'
    'Gradient computation method.'
    parameter_frequencies = [(0.5, 1.0)]
    'Frequencies of the operation parameter with respect to an expectation value.'

    def generator(self):
        w1, w2 = self.wires
        return 0.25 * (qml.X(w1) @ qml.Y(w2) - qml.Y(w1) @ qml.X(w2))

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SingleExcitation.matrix`


        Args:
          phi (tensor_like or float): rotation angle

        Returns:
          tensor_like: canonical matrix

        **Example**

        >>> qml.SingleExcitation.compute_matrix(torch.tensor(0.5))
        tensor([[ 1.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.9689, -0.2474,  0.0000],
                [ 0.0000,  0.2474,  0.9689,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  1.0000]])
        """
        return _single_excitations_matrix(phi, 0.0)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.SingleExcitation.decomposition`.

        Args:
            phi (float): rotation angle :math:`\\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.SingleExcitation.compute_decomposition(1.23, wires=(0,1))
        [Adjoint(T(wires=[0])),
         Hadamard(wires=[0]),
         S(wires=[0]),
         Adjoint(T(wires=[1])),
         Adjoint(S(wires=[1])),
         Hadamard(wires=[1]),
         CNOT(wires=[1, 0]),
         RZ(-0.615, wires=[0]),
         RY(0.615, wires=[1]),
         CNOT(wires=[1, 0]),
         Adjoint(S(wires=[0])),
         Hadamard(wires=[0]),
         T(wires=[0]),
         Hadamard(wires=[1]),
         S(wires=[1]),
         T(wires=[1])]

        """
        decomp_ops = [qml.adjoint(qml.T)(wires=wires[0]), qml.Hadamard(wires=wires[0]), qml.S(wires=wires[0]), qml.adjoint(qml.T)(wires=wires[1]), qml.adjoint(qml.S)(wires=wires[1]), qml.Hadamard(wires=wires[1]), qml.CNOT(wires=[wires[1], wires[0]]), qml.RZ(-phi / 2, wires=wires[0]), qml.RY(phi / 2, wires=wires[1]), qml.CNOT(wires=[wires[1], wires[0]]), qml.adjoint(qml.S)(wires=wires[0]), qml.Hadamard(wires=wires[0]), qml.T(wires=wires[0]), qml.Hadamard(wires=wires[1]), qml.S(wires=wires[1]), qml.T(wires=wires[1])]
        return decomp_ops

    def adjoint(self):
        phi, = self.parameters
        return SingleExcitation(-phi, wires=self.wires)

    def pow(self, z):
        return [SingleExcitation(self.data[0] * z, wires=self.wires)]

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'G', cache=cache)