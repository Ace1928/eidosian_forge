import warnings
from itertools import product
import numpy as np
from scipy.linalg import fractional_matrix_power
from pennylane.math import norm, cast, eye, zeros, transpose, conj, sqrt, sqrt_matrix
from pennylane import numpy as pnp
import pennylane as qml
from pennylane.operation import AnyWires, DecompositionUndefinedError, Operation
from pennylane.wires import Wires
class QubitUnitary(Operation):
    """QubitUnitary(U, wires)
    Apply an arbitrary unitary matrix with a dimension that is a power of two.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (2,)
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
        unitary_check (bool): check for unitarity of the given matrix

    Raises:
        ValueError: if the number of wires doesn't fit the dimensions of the matrix

    **Example**

    >>> dev = qml.device('default.qubit', wires=1)
    >>> U = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.QubitUnitary(U, wires=0)
    ...     return qml.expval(qml.Z(0))
    >>> print(example_circuit())
    0.0
    """
    num_wires = AnyWires
    'int: Number of wires that the operator acts on.'
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (2,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = None
    'Gradient computation method.'

    def __init__(self, U, wires, id=None, unitary_check=False):
        wires = Wires(wires)
        U_shape = qml.math.shape(U)
        dim = 2 ** len(wires)
        if len(U_shape) not in {2, 3} or U_shape[-2:] != (dim, dim):
            raise ValueError(f'Input unitary must be of shape {(dim, dim)} or (batch_size, {dim}, {dim}) to act on {len(wires)} wires.')
        if unitary_check and (not (qml.math.is_abstract(U) or qml.math.allclose(qml.math.einsum('...ij,...kj->...ik', U, qml.math.conj(U)), qml.math.eye(dim), atol=1e-06))):
            warnings.warn(f'Operator {U}\n may not be unitary. Verify unitarity of operation, or use a datatype with increased precision.', UserWarning)
        super().__init__(U, wires=wires, id=id)

    @staticmethod
    def compute_matrix(U):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.QubitUnitary.matrix`

        Args:
            U (tensor_like): unitary matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> U = np.array([[0.98877108+0.j, 0.-0.14943813j], [0.-0.14943813j, 0.98877108+0.j]])
        >>> qml.QubitUnitary.compute_matrix(U)
        [[0.98877108+0.j, 0.-0.14943813j],
        [0.-0.14943813j, 0.98877108+0.j]]
        """
        return U

    @staticmethod
    def compute_decomposition(U, wires):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.

        A decomposition is only defined for matrices that act on either one or two wires. For more
        than two wires, this method raises a ``DecompositionUndefined``.

        See :func:`~.transforms.one_qubit_decomposition` and :func:`~.ops.two_qubit_decomposition`
        for more information on how the decompositions are computed.

        .. seealso:: :meth:`~.QubitUnitary.decomposition`.

        Args:
            U (array[complex]): square unitary matrix
            wires (Iterable[Any] or Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> U = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        >>> qml.QubitUnitary.compute_decomposition(U, 0)
        [Rot(tensor(3.14159265, requires_grad=True), tensor(1.57079633, requires_grad=True), tensor(0., requires_grad=True), wires=[0])]

        """
        shape = qml.math.shape(U)
        is_batched = len(shape) == 3
        shape_without_batch_dim = shape[1:] if is_batched else shape
        if shape_without_batch_dim == (2, 2):
            return qml.ops.one_qubit_decomposition(U, Wires(wires)[0])
        if shape_without_batch_dim == (4, 4):
            if is_batched:
                raise DecompositionUndefinedError('The decomposition of a two-qubit QubitUnitary does not support broadcasting.')
            return qml.ops.two_qubit_decomposition(U, Wires(wires))
        return super(QubitUnitary, QubitUnitary).compute_decomposition(U, wires=wires)

    @property
    def has_decomposition(self):
        return len(self.wires) < 3

    def adjoint(self):
        U = self.matrix()
        return QubitUnitary(qml.math.moveaxis(qml.math.conj(U), -2, -1), wires=self.wires)

    def pow(self, z):
        mat = self.matrix()
        if isinstance(z, int) and qml.math.get_deep_interface(mat) != 'tensorflow':
            pow_mat = qml.math.linalg.matrix_power(mat, z)
        elif self.batch_size is not None or qml.math.shape(z) != ():
            return super().pow(z)
        else:
            pow_mat = qml.math.convert_like(fractional_matrix_power(mat, z), mat)
        return [QubitUnitary(pow_mat, wires=self.wires)]

    def _controlled(self, wire):
        return qml.ControlledQubitUnitary(*self.parameters, control_wires=wire, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'U', cache=cache)