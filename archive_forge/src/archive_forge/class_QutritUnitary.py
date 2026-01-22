import warnings
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires
class QutritUnitary(Operation):
    """Apply an arbitrary, fixed unitary matrix.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        wires(Sequence[int] or int): the wire(s) the operation acts on
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    **Example**

    >>> dev = qml.device('default.qutrit', wires=1)
    >>> U = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.QutritUnitary(U, wires=0)
    ...     return qml.state()
    >>> print(example_circuit())
    [0.70710678+0.j 0.70710678+0.j 0.        +0.j]
    """
    num_wires = AnyWires
    'int: Number of wires that the operator acts on.'
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (2,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = None
    'Gradient computation method.'

    def __init__(self, *params, wires):
        wires = Wires(wires)
        if not isinstance(self, ControlledQutritUnitary):
            U = params[0]
            U_shape = qml.math.shape(U)
            dim = 3 ** len(wires)
            if not (len(U_shape) in {2, 3} and U_shape[-2:] == (dim, dim)):
                raise ValueError(f'Input unitary must be of shape {(dim, dim)} or (batch_size, {dim}, {dim}) to act on {len(wires)} wires.')
            if not (qml.math.is_abstract(U) or qml.math.allclose(qml.math.einsum('...ij,...kj->...ik', U, qml.math.conj(U)), qml.math.eye(dim), atol=1e-06)):
                warnings.warn(f'Operator {U}\n may not be unitary. Verify unitarity of operation, or use a datatype with increased precision.', UserWarning)
        super().__init__(*params, wires=wires)

    @staticmethod
    def compute_matrix(U):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.QutritUnitary.matrix`

        Args:
            U (tensor_like): unitary matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> U = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
        >>> qml.QutritUnitary.compute_matrix(U)
        array([[ 0.70710678,  0.70710678,  0.        ],
               [ 0.70710678, -0.70710678,  0.        ],
               [ 0.        ,  0.        ,  1.        ]])
        """
        return U

    def adjoint(self):
        U = self.matrix()
        return QutritUnitary(qml.math.conj(qml.math.moveaxis(U, -2, -1)), wires=self.wires)

    def pow(self, z):
        if isinstance(z, int):
            return [QutritUnitary(qml.math.linalg.matrix_power(self.matrix(), z), wires=self.wires)]
        return super().pow(z)

    def _controlled(self, wire):
        return ControlledQutritUnitary(*self.parameters, control_wires=wire, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'U', cache=cache)