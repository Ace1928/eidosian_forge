from quantum chemistry applications.
import functools
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane.operation import Operation
class OrbitalRotation(Operation):
    """
    Spin-adapted spatial orbital rotation.

    For two neighbouring spatial orbitals :math:`\\{|\\Phi_{0}\\rangle, |\\Phi_{1}\\rangle\\}`, this operation
    performs the following transformation

    .. math::
        &|\\Phi_{0}\\rangle = \\cos(\\phi/2)|\\Phi_{0}\\rangle - \\sin(\\phi/2)|\\Phi_{1}\\rangle\\\\
        &|\\Phi_{1}\\rangle = \\cos(\\phi/2)|\\Phi_{0}\\rangle + \\sin(\\phi/2)|\\Phi_{1}\\rangle,

    with the same orbital operation applied in the :math:`\\alpha` and :math:`\\beta` spin orbitals.

    .. figure:: ../../_static/qchem/orbital_rotation.jpeg
        :align: center
        :width: 100%
        :target: javascript:void(0);

    Here, :math:`G(\\phi)` represents a single-excitation Givens rotation and :math:`f\\text{SWAP}(\\pi)`
    represents the fermionic swap operator, implemented in PennyLane as the
    :class:`~.SingleExcitation` operation and :class:`~.FermionicSWAP` operation, respectively. This
    implementation is a modified version of the one given in `Anselmetti et al. (2021) <https://doi.org/10.1088/1367-2630/ac2cb3>`__\\ ,
    and is consistent with the Jordan-Wigner mapping in interleaved ordering.

    **Details:**

    * Number of wires: 4
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The ``OrbitalRotation`` operator has 4 equidistant frequencies
      :math:`\\{0.5, 1, 1.5, 2\\}`, and thus permits an 8-term parameter-shift rule.
      (see `Wierichs et al. (2022) <https://doi.org/10.22331/q-2022-03-30-677>`__).

    Args:
        phi (float): rotation angle :math:`\\phi`
        wires (Sequence[int]): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    **Example**

    .. code-block::

        >>> dev = qml.device('default.qubit', wires=4)
        >>> @qml.qnode(dev)
        ... def circuit(phi):
        ...     qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
        ...     qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])
        ...     return qml.state()
        >>> circuit(0.1)
        array([ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                0.00249792+0.j,  0.        +0.j,  0.        +0.j,
                0.04991671+0.j,  0.        +0.j,  0.        +0.j,
               -0.04991671+0.j,  0.        +0.j,  0.        +0.j,
                0.99750208+0.j,  0.        +0.j,  0.        +0.j,
                0.        +0.j])
    """
    num_wires = 4
    'int: Number of wires that the operator acts on.'
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = 'A'
    'Gradient computation method.'
    parameter_frequencies = [(0.5, 1.0, 1.5, 2.0)]
    'Frequencies of the operation parameter with respect to an expectation value.'

    def generator(self):
        w0, w1, w2, w3 = self.wires
        return 0.25 * (qml.X(w0) @ qml.Z(w1) @ qml.Y(w2) - qml.Y(w0) @ qml.Z(w1) @ qml.X(w2) + qml.X(w1) @ qml.Z(w2) @ qml.Y(w3) - qml.Y(w1) @ qml.Z(w2) @ qml.X(w3))

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)
    mask_s = np.zeros((16, 16))
    mask_s[1, 4] = mask_s[2, 8] = mask_s[13, 7] = mask_s[14, 11] = -1
    mask_s[4, 1] = mask_s[8, 2] = mask_s[7, 13] = mask_s[11, 14] = 1
    mask_cs = np.zeros((16, 16))
    mask_cs[6, 3] = mask_cs[3, 9] = mask_cs[12, 6] = mask_cs[9, 12] = -1
    mask_cs[3, 6] = mask_cs[9, 3] = mask_cs[6, 12] = mask_cs[12, 9] = 1
    mask_s2 = np.zeros((16, 16))
    mask_s2[3, 12] = mask_s2[12, 3] = mask_s2[6, 9] = mask_s2[9, 6] = 1

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.OrbitalRotation.matrix`


        Args:
          phi (tensor_like or float): rotation angle

        Returns:
          tensor_like: canonical matrix
        """
        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)
        if qml.math.ndim(phi) == 0:
            diag = qml.math.diag([1.0, c, c, c ** 2, c, 1.0, c ** 2, c, c, c ** 2, 1.0, c, c ** 2, c, c, 1.0])
            if qml.math.get_interface(phi) == 'torch':
                mask_s = qml.math.convert_like(OrbitalRotation.mask_s, phi)
                mask_cs = qml.math.convert_like(OrbitalRotation.mask_cs, phi)
                mask_s2 = qml.math.convert_like(OrbitalRotation.mask_s2, phi)
                return diag + s * mask_s + c * s * mask_cs + s ** 2 * mask_s2
            return diag + s * OrbitalRotation.mask_s + c * s * OrbitalRotation.mask_cs + s ** 2 * OrbitalRotation.mask_s2
        ones = qml.math.ones_like(c)
        diag = qml.math.stack([ones, c, c, c ** 2, c, ones, c ** 2, c, c, c ** 2, ones, c, c ** 2, c, c, ones], axis=-1)
        diag = qml.math.einsum('ij,jk->ijk', diag, I16)
        off_diag = qml.math.einsum('i,jk->ijk', s, OrbitalRotation.mask_s) + qml.math.einsum('i,jk->ijk', c * s, OrbitalRotation.mask_cs) + qml.math.einsum('i,jk->ijk', s ** 2, OrbitalRotation.mask_s2)
        return diag + off_diag

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.OrbitalRotation.decomposition`.

        This operator is decomposed into two :class:`~.SingleExcitation` gates. For a decomposition
        into more elementary gates, see page 18 of
        `"Local, Expressive, Quantum-Number-Preserving VQE Ansatze for Fermionic Systems" <https://doi.org/10.1088/1367-2630/ac2cb3>`_ .

        Args:
            phi (float): rotation angle :math:`\\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.OrbitalRotation.compute_decomposition(1.2, wires=[0, 1, 2, 3])
        [qml.FermionicSWAP(np.pi, wires=[1, 2]), SingleExcitation(1.2, wires=[0, 2]),
         SingleExcitation(1.2, wires=[1, 3]), qml.FermionicSWAP(np.pi, wires=[1, 2])]

        """
        return [qml.FermionicSWAP(np.pi, wires=[wires[1], wires[2]]), qml.SingleExcitation(phi, wires=[wires[0], wires[1]]), qml.SingleExcitation(phi, wires=[wires[2], wires[3]]), qml.FermionicSWAP(np.pi, wires=[wires[1], wires[2]])]

    def adjoint(self):
        phi, = self.parameters
        return OrbitalRotation(-phi, wires=self.wires)