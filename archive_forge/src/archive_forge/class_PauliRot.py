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
class PauliRot(Operation):
    """
    Arbitrary Pauli word rotation.

    .. math::

        RP(\\theta, P) = \\exp(-i \\frac{\\theta}{2} P)

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\\frac{d}{d\\theta}f(RP(\\theta)) = \\frac{1}{2}\\left[f(RP(\\theta +\\pi/2)) - f(RP(\\theta-\\pi/2))\\right]`
      where :math:`f` is an expectation value depending on :math:`RP(\\theta)`.

    .. note::

        If the ``PauliRot`` gate is not supported on the targeted device, PennyLane
        will decompose the gate using :class:`~.RX`, :class:`~.Hadamard`, :class:`~.RZ`
        and :class:`~.CNOT` gates.

    Args:
        theta (float): rotation angle :math:`\\theta`
        pauli_word (string): the Pauli word defining the rotation
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    **Example**

    >>> dev = qml.device('default.qubit', wires=1)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.PauliRot(0.5, 'X',  wires=0)
    ...     return qml.expval(qml.Z(0))
    >>> print(example_circuit())
    0.8775825618903724
    """
    num_wires = AnyWires
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    do_check_domain = False
    grad_method = 'A'
    parameter_frequencies = [(1,)]
    _ALLOWED_CHARACTERS = 'IXYZ'
    _PAULI_CONJUGATION_MATRICES = {'X': Hadamard.compute_matrix(), 'Y': RX.compute_matrix(np.pi / 2), 'Z': np.array([[1, 0], [0, 1]])}

    def __init__(self, theta, pauli_word, wires=None, id=None):
        super().__init__(theta, wires=wires, id=id)
        self.hyperparameters['pauli_word'] = pauli_word
        if not PauliRot._check_pauli_word(pauli_word):
            raise ValueError(f'The given Pauli word "{pauli_word}" contains characters that are not allowed. Allowed characters are I, X, Y and Z')
        num_wires = 1 if isinstance(wires, int) else len(wires)
        if not len(pauli_word) == num_wires:
            raise ValueError(f'The given Pauli word has length {len(pauli_word)}, length {num_wires} was expected for wires {wires}')

    def __repr__(self):
        return f'PauliRot({self.data[0]}, {self.hyperparameters['pauli_word']}, wires={self.wires.tolist()})'

    def label(self, decimals=None, base_label=None, cache=None):
        """A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that caries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        **Example:**

        >>> op = qml.PauliRot(0.1, "XYY", wires=(0,1,2))
        >>> op.label()
        'RXYY'
        >>> op.label(decimals=2)
        'RXYY\\n(0.10)'
        >>> op.label(base_label="PauliRot")
        'PauliRot\\n(0.10)'

        """
        pauli_word = self.hyperparameters['pauli_word']
        op_label = base_label or 'R' + pauli_word
        if decimals is not None and self.batch_size is None:
            param_string = f'\n({qml.math.asarray(self.parameters[0]):.{decimals}f})'
            op_label += param_string
        return op_label

    @staticmethod
    def _check_pauli_word(pauli_word):
        """Check that the given Pauli word has correct structure.

        Args:
            pauli_word (str): Pauli word to be checked

        Returns:
            bool: Whether the Pauli word has correct structure.
        """
        return all((pauli in PauliRot._ALLOWED_CHARACTERS for pauli in set(pauli_word)))

    @staticmethod
    def compute_matrix(theta, pauli_word):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.PauliRot.matrix`


        Args:
            theta (tensor_like or float): rotation angle
            pauli_word (str): string representation of Pauli word

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.PauliRot.compute_matrix(0.5, 'X')
        [[9.6891e-01+4.9796e-18j 2.7357e-17-2.4740e-01j]
         [2.7357e-17-2.4740e-01j 9.6891e-01+4.9796e-18j]]
        """
        if not PauliRot._check_pauli_word(pauli_word):
            raise ValueError(f'The given Pauli word "{pauli_word}" contains characters that are not allowed. Allowed characters are I, X, Y and Z')
        interface = qml.math.get_interface(theta)
        if interface == 'tensorflow':
            theta = qml.math.cast_like(theta, 1j)
        if set(pauli_word) == {'I'}:
            exp = qml.math.exp(-0.5j * theta)
            iden = qml.math.eye(2 ** len(pauli_word), like=theta)
            if qml.math.get_interface(theta) == 'tensorflow':
                iden = qml.math.cast_like(iden, 1j)
            if qml.math.get_interface(theta) == 'torch':
                td = exp.device
                iden = iden.to(td)
            if qml.math.ndim(theta) == 0:
                return exp * iden
            return qml.math.stack([e * iden for e in exp])
        non_identity_wires, non_identity_gates = zip(*[(wire, gate) for wire, gate in enumerate(pauli_word) if gate != 'I'])
        multi_Z_rot_matrix = MultiRZ.compute_matrix(theta, len(non_identity_gates))
        conjugation_matrix = functools.reduce(qml.math.kron, [PauliRot._PAULI_CONJUGATION_MATRICES[gate] for gate in non_identity_gates])
        if interface == 'tensorflow':
            conjugation_matrix = qml.math.cast_like(conjugation_matrix, 1j)
        return expand_matrix(qml.math.einsum('...jk,ij->...ik', qml.math.tensordot(multi_Z_rot_matrix, conjugation_matrix, axes=[[-1], [0]]), qml.math.conj(conjugation_matrix)), non_identity_wires, list(range(len(pauli_word))))

    def generator(self):
        pauli_word = self.hyperparameters['pauli_word']
        wire_map = {w: i for i, w in enumerate(self.wires)}
        return -0.5 * qml.pauli.string_to_pauli_word(pauli_word, wire_map=wire_map)

    @staticmethod
    def compute_eigvals(theta, pauli_word):
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.PauliRot.eigvals`


        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.PauliRot.compute_eigvals(torch.tensor(0.5), "X")
        tensor([0.9689-0.2474j, 0.9689+0.2474j])
        """
        if qml.math.get_interface(theta) == 'tensorflow':
            theta = qml.math.cast_like(theta, 1j)
        if set(pauli_word) == {'I'}:
            exp = qml.math.exp(-0.5j * theta)
            ones = qml.math.ones(2 ** len(pauli_word), like=theta)
            if qml.math.get_interface(theta) == 'tensorflow':
                ones = qml.math.cast_like(ones, 1j)
            if qml.math.ndim(theta) == 0:
                return exp * ones
            return qml.math.tensordot(exp, ones, axes=0)
        return MultiRZ.compute_eigvals(theta, len(pauli_word))

    @staticmethod
    def compute_decomposition(theta, wires, pauli_word):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.PauliRot.decomposition`.

        Args:
            theta (float): rotation angle :math:`\\theta`
            wires (Iterable, Wires): the wires the operation acts on
            pauli_word (string): the Pauli word defining the rotation

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.PauliRot.compute_decomposition(1.2, "XY", wires=(0,1))
        [Hadamard(wires=[0]),
        RX(1.5707963267948966, wires=[1]),
        MultiRZ(1.2, wires=[0, 1]),
        Hadamard(wires=[0]),
        RX(-1.5707963267948966, wires=[1])]

        """
        if isinstance(wires, int):
            wires = [wires]
        if set(pauli_word) == {'I'}:
            return []
        active_wires, active_gates = zip(*[(wire, gate) for wire, gate in zip(wires, pauli_word) if gate != 'I'])
        ops = []
        for wire, gate in zip(active_wires, active_gates):
            if gate == 'X':
                ops.append(Hadamard(wires=[wire]))
            elif gate == 'Y':
                ops.append(RX(np.pi / 2, wires=[wire]))
        ops.append(MultiRZ(theta, wires=list(active_wires)))
        for wire, gate in zip(active_wires, active_gates):
            if gate == 'X':
                ops.append(Hadamard(wires=[wire]))
            elif gate == 'Y':
                ops.append(RX(-np.pi / 2, wires=[wire]))
        return ops

    def adjoint(self):
        return PauliRot(-self.parameters[0], self.hyperparameters['pauli_word'], wires=self.wires)

    def pow(self, z):
        return [PauliRot(self.data[0] * z, self.hyperparameters['pauli_word'], wires=self.wires)]