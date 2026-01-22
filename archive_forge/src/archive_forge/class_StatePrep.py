import numpy as np
from pennylane import math
from pennylane.operation import AnyWires, Operation, StatePrepBase
from pennylane.templates.state_preparations import BasisStatePreparation, MottonenStatePreparation
from pennylane.wires import Wires, WireError
class StatePrep(StatePrepBase):
    """StatePrep(state, wires)
    Prepare subsystems using the given ket vector in the computational basis.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    .. note::

        If the ``StatePrep`` operation is not supported natively on the
        target device, PennyLane will attempt to decompose the operation
        using the method developed by Möttönen et al. (Quantum Info. Comput.,
        2005).

    .. note::

        When called in the middle of a circuit, the action of the operation is defined
        as :math:`U|0\\rangle = |\\psi\\rangle`

    Args:
        state (array[complex]): a state vector of size 2**len(wires)
        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    **Example**

    >>> dev = qml.device('default.qubit', wires=2)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.StatePrep(np.array([1, 0, 0, 0]), wires=range(2))
    ...     return qml.state()
    >>> print(example_circuit())
    [1.+0.j 0.+0.j 0.+0.j 0.+0.j]
    """
    num_wires = AnyWires
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (1,)
    'int: Number of dimensions per trainable parameter of the operator.'

    def __init__(self, state, wires, id=None):
        super().__init__(state, wires=wires, id=id)
        state = self.parameters[0]
        if len(state.shape) == 1:
            state = math.reshape(state, (1, state.shape[0]))
        if state.shape[1] != 2 ** len(self.wires):
            raise ValueError('State vector must have shape (2**wires,) or (batch_size, 2**wires).')
        param = math.cast(state, np.complex128)
        if not math.is_abstract(param):
            norm = math.linalg.norm(param, axis=-1, ord=2)
            if not math.allclose(norm, 1.0, atol=1e-10):
                raise ValueError('Sum of amplitudes-squared does not equal one.')

    @staticmethod
    def compute_decomposition(state, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.StatePrep.decomposition`.

        Args:
            state (array[complex]): a state vector of size 2**len(wires)
            wires (Iterable, Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.StatePrep.compute_decomposition(np.array([1, 0, 0, 0]), wires=range(2))
        [MottonenStatePreparation(tensor([1, 0, 0, 0], requires_grad=True), wires=[0, 1])]

        """
        return [MottonenStatePreparation(state, wires)]

    def state_vector(self, wire_order=None):
        num_op_wires = len(self.wires)
        op_vector_shape = (-1,) + (2,) * num_op_wires if self.batch_size else (2,) * num_op_wires
        op_vector = math.reshape(self.parameters[0], op_vector_shape)
        if wire_order is None or Wires(wire_order) == self.wires:
            return op_vector
        wire_order = Wires(wire_order)
        if not wire_order.contains_wires(self.wires):
            raise WireError(f'Custom wire_order must contain all {self.name} wires')
        extra_wires = Wires(set(wire_order) - set(self.wires))
        for _ in extra_wires:
            op_vector = math.stack([op_vector, math.zeros_like(op_vector)], axis=-1)
        current_wires = self.wires + extra_wires
        transpose_axes = [current_wires.index(w) for w in wire_order]
        if self.batch_size:
            transpose_axes = [0] + [a + 1 for a in transpose_axes]
        return math.transpose(op_vector, transpose_axes)