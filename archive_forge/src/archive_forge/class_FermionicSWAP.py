from quantum chemistry applications.
import functools
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane.operation import Operation
class FermionicSWAP(Operation):
    """Fermionic SWAP rotation.

    .. math:: U(\\phi) = \\begin{bmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & e^{i \\phi/2} \\cos(\\phi/2) & -ie^{i \\phi/2} \\sin(\\phi/2) & 0 \\\\
                0 & -ie^{i \\phi/2} \\sin(\\phi/2) & e^{i \\phi/2} \\cos(\\phi/2) & 0 \\\\
                0 & 0 & 0 & e^{i \\phi}
            \\end{bmatrix}.

    This operation performs a rotation in the adjacent fermionic modes under the Jordan-Wigner mapping,
    and is realized by the following transformation of basis states:

    .. math::
        &|00\\rangle \\mapsto |00\\rangle\\\\
        &|01\\rangle \\mapsto e^{i \\phi/2} \\cos(\\phi/2)|01\\rangle - ie^{i \\phi/2} \\sin(\\phi/2)|10\\rangle\\\\
        &|10\\rangle \\mapsto -ie^{i \\phi/2} \\sin(\\phi/2)|01\\rangle + e^{i \\phi/2} \\cos(\\phi/2)|10\\rangle\\\\
        &|11\\rangle \\mapsto e^{i \\phi}|11\\rangle,

    where qubits in :math:`|0\\rangle` and :math:`|1\\rangle` states represent a hole and a fermion in
    the orbital, respectively. It preserves anti-symmetrization of orbitals by applying a phase factor
    of :math:`e^{i \\phi/2}` to the state for each qubit initially in :math:`|1\\rangle` state. Consequently,
    for :math:`\\phi=\\pi`, the given rotation will essentially perform a SWAP operation on the qubits while
    applying a global phase of :math:`-1`, if both qubits are :math:`|1\\rangle`.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\\frac{d}{d\\phi}f(U(\\phi)) = \\frac{1}{2}\\left[f(U(\\phi+\\pi/2)) - f(U(\\phi-\\pi/2))\\right]`
      where :math:`f` is an expectation value depending on :math:`U(\\phi)`

    Args:
        phi (float): rotation angle :math:`\\phi`
        wires (Sequence[int]): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    **Example**

    The following circuit performs the transformation: :math:`|01\\rangle \\mapsto e^{i \\phi/2}
    \\cos(\\phi/2)|01\\rangle - ie^{i \\phi/2} \\sin(\\phi/2)|10\\rangle`, where :math:`\\phi=0.1`:

    .. code-block::

        >>> dev = qml.device('default.qubit', wires=2)
        >>> @qml.qnode(dev)
        ... def circuit(phi):
        ...     qml.X(1)
        ...     qml.FermionicSWAP(phi, wires=[0, 1])
        ...     return qml.state()
        >>> circuit(0.1)
        array([0.+0.j, 0.9975+0.04992j, 0.0025-0.04992j, 0.+0.j])
    """
    num_wires = 2
    'int: Number of wires that the operator acts on.'
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = 'A'
    'Gradient computation method.'
    parameter_frequencies = [(1,)]
    'Frequencies of the operation parameter with respect to an expectation value.'

    def generator(self):
        w1, w2 = self.wires
        return 0.5 * qml.Identity(w1) @ qml.Identity(w2) - 0.25 * (qml.Identity(w1) @ qml.Z(w2) + qml.Z(w1) @ qml.Identity(w2) + qml.X(w1) @ qml.X(w2) + qml.Y(w1) @ qml.Y(w2))

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.FermionicSWAP.matrix`


        Args:
          phi (tensor_like or float): rotation angle

        Returns:
          tensor_like: canonical matrix

        **Example**

        >>> qml.FermionicSWAP.compute_matrix(torch.tensor(0.5))
        tensor([1.   +0.j, 0.   +0.j  , 0.   +0.j  , 0.   +0.j   ],
               [0.   +0.j, 0.939+0.24j, 0.061-0.24j, 0.   +0.j   ],
               [0.   +0.j, 0.061-0.24j, 0.939+0.24j, 0.   +0.j   ],
               [0.   +0.j, 0.   +0.j  , 0.   +0.j  , 0.878+0.479j]])
        """
        if qml.math.get_interface(phi) == 'tensorflow':
            phi = qml.math.cast_like(phi, 1j)
        c = qml.math.cast_like(qml.math.cos(phi / 2), 1j)
        s = qml.math.cast_like(qml.math.sin(phi / 2), 1j)
        g = qml.math.cast_like(qml.math.exp(1j * phi / 2), 1j)
        p = qml.math.cast_like(qml.math.exp(1j * phi), 1j)
        zeros = qml.math.zeros_like(phi)
        ones = qml.math.ones_like(phi)
        rows = [[ones, zeros, zeros, zeros], [zeros, g * c, -1j * g * s, zeros], [zeros, -1j * g * s, g * c, zeros], [zeros, zeros, zeros, p]]
        return qml.math.stack([stack_last(row) for row in rows], axis=-2)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.FermionicSWAP.decomposition`.

        Args:
            phi (float): rotation angle :math:`\\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.FermionicSWAP.compute_decomposition(0.2, wires=(0, 1))
        [Hadamard(wires=[0]),
         Hadamard(wires=[1]),
         MultiRZ(0.1, wires=[0, 1]),
         Hadamard(wires=[0]),
         Hadamard(wires=[1]),
         RX(1.5707963267948966, wires=[0]),
         RX(1.5707963267948966, wires=[1]),
         MultiRZ(0.1, wires=[0, 1]),
         RX(-1.5707963267948966, wires=[0]),
         RX(-1.5707963267948966, wires=[1]),
         RZ(0.1, wires=[0]),
         RZ(0.1, wires=[1]),
         Exp(0.1j Identity)]
        """
        decomp_ops = [qml.Hadamard(wires=wires[0]), qml.Hadamard(wires=wires[1]), qml.MultiRZ(phi / 2, wires=[wires[0], wires[1]]), qml.Hadamard(wires=wires[0]), qml.Hadamard(wires=wires[1]), qml.RX(np.pi / 2, wires=wires[0]), qml.RX(np.pi / 2, wires=wires[1]), qml.MultiRZ(phi / 2, wires=[wires[0], wires[1]]), qml.RX(-np.pi / 2, wires=wires[0]), qml.RX(-np.pi / 2, wires=wires[1]), qml.RZ(phi / 2, wires=wires[0]), qml.RZ(phi / 2, wires=wires[1]), qml.exp(qml.Identity(wires=[wires[0], wires[1]]), 0.5j * phi)]
        return decomp_ops

    def adjoint(self):
        phi, = self.parameters
        return FermionicSWAP(-phi, wires=self.wires)

    def pow(self, z):
        return [FermionicSWAP(self.data[0] * z, wires=self.wires)]

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'fSWAP', cache=cache)