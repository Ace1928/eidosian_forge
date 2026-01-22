from abc import abstractmethod
from copy import copy
import numpy as np
import pennylane as qml
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.queuing import QueuingManager
class SymbolicOp(Operator):
    """Developer-facing base class for single-operator symbolic operators.

    Args:
        base (~.operation.Operator): the base operation that is modified symbolicly
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified

    This *developer-facing* class can serve as a parent to single base symbolic operators, such as
    :class:`~.ops.op_math.Adjoint`.

    New symbolic operators can inherit from this class to receive some common default behavior, such
    as deferring properties to the base class, copying the base class during a shallow copy, and
    updating the metadata of the base operator during queueing.

    The child symbolic operator should define the `_name` property during initialization and define
    any relevant representations, such as :meth:`~.operation.Operator.matrix`,
    :meth:`~.operation.Operator.diagonalizing_gates`, :meth:`~.operation.Operator.eigvals`, and
    :meth:`~.operation.Operator.decomposition`.
    """
    _name = 'Symbolic'

    def __copy__(self):
        copied_op = object.__new__(type(self))
        for attr, value in vars(self).items():
            if attr not in {'_hyperparameters'}:
                setattr(copied_op, attr, value)
        copied_op._hyperparameters = copy(self.hyperparameters)
        copied_op.hyperparameters['base'] = copy(self.base)
        return copied_op

    def __init__(self, base, id=None):
        self.hyperparameters['base'] = base
        self._id = id
        self.queue_idx = None
        self._pauli_rep = None
        self.queue()

    @property
    def batch_size(self):
        return self.base.batch_size

    @property
    def base(self) -> Operator:
        """The base operator."""
        return self.hyperparameters['base']

    @property
    def data(self):
        """The trainable parameters"""
        return self.base.data

    @data.setter
    def data(self, new_data):
        self.base.data = new_data

    @property
    def num_params(self):
        return self.base.num_params

    @property
    def wires(self):
        return self.base.wires

    @property
    def basis(self):
        return self.base.basis

    @property
    def num_wires(self):
        """Number of wires the operator acts on."""
        return len(self.wires)

    @property
    def has_matrix(self):
        return self.base.has_matrix

    @property
    def is_hermitian(self):
        return self.base.is_hermitian

    @property
    def _queue_category(self):
        return self.base._queue_category

    def queue(self, context=QueuingManager):
        context.remove(self.base)
        context.append(self)
        return self

    @property
    def arithmetic_depth(self) -> int:
        return 1 + self.base.arithmetic_depth

    @property
    def hash(self):
        return hash((str(self.name), self.base.hash))

    def map_wires(self, wire_map: dict):
        new_op = copy(self)
        new_op.hyperparameters['base'] = self.base.map_wires(wire_map=wire_map)
        if (p_rep := new_op.pauli_rep) is not None:
            new_op._pauli_rep = p_rep.map_wires(wire_map)
        return new_op