from copy import copy
import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires
from pennylane.ops import Identity
class IntegerComparator(Operation):
    """IntegerComparator(value, geq, wires)
    Apply a controlled Pauli X gate using integer comparison as the condition.

    Given a basis state :math:`\\vert n \\rangle`, where :math:`n` is a positive integer, and a fixed positive
    integer :math:`L`, flip a target qubit if :math:`n \\geq L`. Alternatively, the flipping condition can
    be :math:`n < L`.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    .. note::

        This operation has one parameter: ``value``. However, ``value`` is simply an integer that is required to define
        the condition upon which a Pauli X gate is applied to the target wire. Given that, IntegerComparator has a
        gradient of zero; ``value`` is a non-differentiable parameter.

    Args:
        value (int): The value :math:`L` that the state's decimal representation is compared against.
        geq (bool): If set to ``True``, the comparison made will be :math:`n \\geq L`. If ``False``, the comparison
            made will be :math:`n < L`.
        wires (Union[Wires, Sequence[int], or int]): Control wire(s) followed by a single target wire where
            the operation acts on.

    **Example**

    >>> dev = qml.device("default.qubit", wires=3)
    >>> @qml.qnode(dev)
    ... def circuit(state, value, geq):
    ...     qml.BasisState(np.array(state), wires=range(3))
    ...     qml.IntegerComparator(value, geq=geq, wires=range(3))
    ...     return qml.state()
    >>> circuit([1, 0, 1], 1, True).reshape(2, 2, 2)[1, 0, 0]
    tensor(1.+0.j, requires_grad=True)
    >>> circuit([0, 1, 0], 3, False).reshape(2, 2, 2)[0, 1, 1]
    tensor(1.+0.j, requires_grad=True)
    """
    is_self_inverse = True
    num_wires = AnyWires
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'
    grad_method = None

    def _flatten(self):
        hp = self.hyperparameters
        metadata = (('work_wires', hp['work_wires']), ('value', hp['value']), ('geq', hp['geq']))
        return (tuple(), (hp['control_wires'] + hp['target_wires'], metadata))

    def __init__(self, value, geq=True, wires=None, work_wires=None):
        if not isinstance(value, int):
            raise ValueError(f'The compared value must be an int. Got {type(value)}.')
        if wires is None:
            raise ValueError('Must specify wires that the operation acts on.')
        if len(wires) > 1:
            control_wires = Wires(wires[:-1])
            wires = Wires(wires[-1])
        else:
            raise ValueError(f'IntegerComparator: wrong number of wires. {len(wires)} wire(s) given. Need at least 2.')
        work_wires = Wires([]) if work_wires is None else Wires(work_wires)
        total_wires = control_wires + wires
        if Wires.shared_wires([total_wires, work_wires]):
            raise ValueError('The work wires must be different from the control and target wires')
        self.hyperparameters['control_wires'] = control_wires
        self.hyperparameters['target_wires'] = wires
        self.hyperparameters['work_wires'] = work_wires
        self.hyperparameters['value'] = value
        self.hyperparameters['geq'] = geq
        self.geq = geq
        self.value = value
        super().__init__(wires=total_wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or f'>={self.value}' if self.geq else f'<{self.value}'

    @staticmethod
    def compute_matrix(value=None, control_wires=None, geq=True, **kwargs):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.IntegerComparator.matrix`

        Args:
            value (int): The value :math:`L` that the state's decimal representation is compared against.
            control_wires (Union[Wires, Sequence[int], or int]): wires to place controls on
            geq (bool): If set to `True`, the comparison made will be :math:`n \\geq L`. If `False`, the comparison
                made will be :math:`n < L`.

        Returns:
           tensor_like: matrix representation

        **Example**

        >>> print(qml.IntegerComparator.compute_matrix(2, [0, 1]))
        [[1. 0. 0. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 1.]
         [0. 0. 0. 0. 0. 0. 1. 0.]]
        >>> print(qml.IntegerComparator.compute_matrix(2, [0, 1], geq=False))
        [[0. 1. 0. 0. 0. 0. 0. 0.]
         [1. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 0. 0. 1.]]
        """
        if value is None:
            raise ValueError('The value to compare to must be specified.')
        if control_wires is None:
            raise ValueError('Must specify the control wires.')
        if not isinstance(value, int):
            raise ValueError(f'The compared value must be an int. Got {type(value)}.')
        small_val = not geq and value == 0
        large_val = geq and value > 2 ** len(control_wires) - 1
        if small_val or large_val:
            mat = np.eye(2 ** (len(control_wires) + 1))
        else:
            values = range(value, 2 ** len(control_wires)) if geq else range(value)
            binary = '0' + str(len(control_wires)) + 'b'
            control_values_list = [format(n, binary) for n in values]
            mat = np.eye(2 ** (len(control_wires) + 1))
            for control_values in control_values_list:
                mat = mat @ qml.MultiControlledX.compute_matrix(control_wires, control_values=control_values)
        return mat

    @staticmethod
    def compute_decomposition(value, geq=True, wires=None, work_wires=None, **kwargs):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.

        .. seealso:: :meth:`~.IntegerComparator.decomposition`.

        Args:
            value (int): The value :math:`L` that the state's decimal representation is compared against.
            geq (bool): If set to ``True``, the comparison made will be :math:`n \\geq L`. If ``False``, the comparison
                made will be :math:`n < L`.
            wires (Union[Wires, Sequence[int], or int]): Control wire(s) followed by a single target wire where
                the operation acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.IntegerComparator.compute_decomposition(4, wires=[0, 1, 2, 3]))
        [MultiControlledX(wires=[0, 1, 2, 3], control_values="100"),
         MultiControlledX(wires=[0, 1, 2, 3], control_values="101"),
         MultiControlledX(wires=[0, 1, 2, 3], control_values="110"),
         MultiControlledX(wires=[0, 1, 2, 3], control_values="111")]
        """
        if not isinstance(value, int):
            raise ValueError(f'The compared value must be an int. Got {type(value)}.')
        if wires is None:
            raise ValueError('Must specify the wires that the operation acts on.')
        if len(wires) > 1:
            control_wires = Wires(wires[:-1])
            wires = Wires(wires[-1])
        else:
            raise ValueError(f'IntegerComparator: wrong number of wires. {len(wires)} wire(s) given. Need at least 2.')
        small_val = not geq and value == 0
        large_val = geq and value > 2 ** len(control_wires) - 1
        if small_val or large_val:
            gates = [Identity(0)]
        else:
            binary = '0' + str(len(control_wires)) + 'b'
            values = range(value, 2 ** len(control_wires)) if geq else range(value)
            binary = '0' + str(len(control_wires)) + 'b'
            control_values_list = [format(n, binary) for n in values]
            gates = []
            for control_values in control_values_list:
                gates.append(qml.MultiControlledX(wires=control_wires + wires, control_values=control_values, work_wires=work_wires))
        return gates

    @property
    def control_wires(self):
        return self.wires[:~0]

    def adjoint(self):
        return copy(self).queue()

    def pow(self, z):
        return super().pow(z % 2)