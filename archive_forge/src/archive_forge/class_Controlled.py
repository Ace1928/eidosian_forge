import warnings
import functools
from copy import copy
from functools import wraps
from inspect import signature
from typing import List
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane import operation
from pennylane import math as qmlmath
from pennylane.operation import Operator
from pennylane.wires import Wires
from pennylane.compiler import compiler
from .symbolicop import SymbolicOp
from .controlled_decompositions import ctrl_decomp_bisect, ctrl_decomp_zyz
class Controlled(SymbolicOp):
    """Symbolic operator denoting a controlled operator.

    Args:
        base (~.operation.Operator): the operator that is controlled
        control_wires (Any): The wires to control on.

    Keyword Args:
        control_values (Iterable[Bool]): The values to control on. Must be the same
            length as ``control_wires``. Defaults to ``True`` for all control wires.
            Provided values are converted to `Bool` internally.
        work_wires (Any): Any auxiliary wires that can be used in the decomposition

    .. note::
        This class, ``Controlled``, denotes a controlled version of any individual operation.
        :class:`~.ControlledOp` adds :class:`~.Operation` specific methods and properties to the
        more general ``Controlled`` class.

    .. seealso:: :class:`~.ControlledOp`, and :func:`~.ctrl`

    **Example**

    >>> base = qml.RX(1.234, 1)
    >>> Controlled(base, (0, 2, 3), control_values=[True, False, True])
    Controlled(RX(1.234, wires=[1]), control_wires=[0, 2, 3], control_values=[True, False, True])
    >>> op = Controlled(base, 0, control_values=[0])
    >>> op
    Controlled(RX(1.234, wires=[1]), control_wires=[0], control_values=[0])

    The operation has both standard :class:`~.operation.Operator` properties
    and ``Controlled`` specific properties:

    >>> op.base
    RX(1.234, wires=[1])
    >>> op.data
    (1.234,)
    >>> op.wires
    <Wires = [0, 1]>
    >>> op.control_wires
    <Wires = [0]>
    >>> op.target_wires
    <Wires = [1]>

    Control values are lists of booleans, indicating whether or not to control on the
    ``0==False`` value or the ``1==True`` wire.

    >>> op.control_values
    [0]

    Provided control values are converted to booleans internally, so
    any "truthy" or "falsy" objects work.

    >>> Controlled(base, ("a", "b", "c"), control_values=["", None, 5]).control_values
    [False, False, True]

    Representations for an operator are available if the base class defines them.
    Sparse matrices are available if the base class defines either a sparse matrix
    or only a dense matrix.

    >>> np.set_printoptions(precision=4) # easier to read the matrix
    >>> qml.matrix(op)
    array([[0.8156+0.j    , 0.    -0.5786j, 0.    +0.j    , 0.    +0.j    ],
           [0.    -0.5786j, 0.8156+0.j    , 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 1.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 1.    +0.j    ]])
    >>> qml.eigvals(op)
    array([1.    +0.j    , 1.    +0.j    , 0.8156+0.5786j, 0.8156-0.5786j])
    >>> print(qml.generator(op, format='observable'))
    (-0.5) [Projector0 X1]
    >>> op.sparse_matrix()
    <4x4 sparse matrix of type '<class 'numpy.complex128'>'
                with 6 stored elements in Compressed Sparse Row format>

    If the provided base matrix is an :class:`~.operation.Operation`, then the created
    object will be of type :class:`~.ops.op_math.ControlledOp`. This class adds some additional
    methods and properties to the basic :class:`~.ops.op_math.Controlled` class.

    >>> type(op)
    <class 'pennylane.ops.op_math.controlled_class.ControlledOp'>
    >>> op.parameter_frequencies
    [(0.5, 1.0)]

    """

    def _flatten(self):
        return ((self.base,), (self.control_wires, tuple(self.control_values), self.work_wires))

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0], control_wires=metadata[0], control_values=metadata[1], work_wires=metadata[2])

    @operation.classproperty
    def __signature__(cls):
        sig = signature(cls.__init__)
        new_parameters = tuple(sig.parameters.values())[1:]
        new_sig = sig.replace(parameters=new_parameters)
        return new_sig

    def __new__(cls, base, *_, **__):
        """If base is an ``Operation``, then a ``ControlledOp`` should be used instead."""
        if isinstance(base, operation.Operation):
            return object.__new__(ControlledOp)
        return object.__new__(Controlled)

    def __init__(self, base, control_wires, control_values=None, work_wires=None, id=None):
        control_wires = Wires(control_wires)
        work_wires = Wires([]) if work_wires is None else Wires(work_wires)
        if control_values is None:
            control_values = [True] * len(control_wires)
        else:
            control_values = [bool(control_values)] if isinstance(control_values, int) else [bool(control_value) for control_value in control_values]
            if len(control_values) != len(control_wires):
                raise ValueError('control_values should be the same length as control_wires')
        if len(Wires.shared_wires([base.wires, control_wires])) != 0:
            raise ValueError('The control wires must be different from the base operation wires.')
        if len(Wires.shared_wires([work_wires, base.wires + control_wires])) != 0:
            raise ValueError('Work wires must be different the control_wires and base operation wires.')
        self.hyperparameters['control_wires'] = control_wires
        self.hyperparameters['control_values'] = control_values
        self.hyperparameters['work_wires'] = work_wires
        self._name = f'C({base.name})'
        super().__init__(base, id)

    @property
    def hash(self):
        if self.base.name in ('RX', 'RY', 'RZ', 'Rot'):
            base_params = str([id(d) if qml.math.is_abstract(d) else qml.math.round(qml.math.real(d) % (4 * np.pi), 10) for d in self.base.data])
            base_hash = hash((str(self.base.name), tuple(self.base.wires.tolist()), base_params))
        else:
            base_hash = self.base.hash
        return hash(('Controlled', base_hash, tuple(self.control_wires.tolist()), tuple(self.control_values), tuple(self.work_wires.tolist())))

    @property
    def has_matrix(self):
        return self.base.has_matrix

    @property
    def batch_size(self):
        return self.base.batch_size

    @property
    def ndim_params(self):
        return self.base.ndim_params

    @property
    def control_values(self):
        """Iterable[Bool]. For each control wire, denotes whether to control on ``True`` or
        ``False``."""
        return self.hyperparameters['control_values']

    @property
    def _control_int(self):
        """Int. Conversion of ``control_values`` to an integer."""
        return sum((2 ** i for i, val in enumerate(reversed(self.control_values)) if val))

    @property
    def control_wires(self):
        """The control wires."""
        return self.hyperparameters['control_wires']

    @property
    def target_wires(self):
        """The wires of the target operator."""
        return self.base.wires

    @property
    def work_wires(self):
        """Additional wires that can be used in the decomposition. Not modified by the operation."""
        return self.hyperparameters['work_wires']

    @property
    def active_wires(self):
        """Wires modified by the operator. This is the control wires followed by the target wires."""
        return self.control_wires + self.target_wires

    @property
    def wires(self):
        return self.control_wires + self.target_wires + self.work_wires

    def map_wires(self, wire_map: dict):
        new_base = self.base.map_wires(wire_map=wire_map)
        new_control_wires = Wires([wire_map.get(wire, wire) for wire in self.control_wires])
        new_work_wires = Wires([wire_map.get(wire, wire) for wire in self.work_wires])
        return ctrl(op=new_base, control=new_control_wires, control_values=self.control_values, work_wires=new_work_wires)

    def __repr__(self):
        params = [f'control_wires={self.control_wires.tolist()}']
        if self.work_wires:
            params.append(f'work_wires={self.work_wires.tolist()}')
        if self.control_values and (not all(self.control_values)):
            params.append(f'control_values={self.control_values}')
        return f'Controlled({self.base}, {', '.join(params)})'

    def label(self, decimals=None, base_label=None, cache=None):
        return self.base.label(decimals=decimals, base_label=base_label, cache=cache)

    def _compute_matrix_from_base(self):
        base_matrix = self.base.matrix()
        interface = qmlmath.get_interface(base_matrix)
        num_target_states = 2 ** len(self.target_wires)
        num_control_states = 2 ** len(self.control_wires)
        total_matrix_size = num_control_states * num_target_states
        padding_left = self._control_int * num_target_states
        padding_right = total_matrix_size - padding_left - num_target_states
        left_pad = qmlmath.convert_like(qmlmath.cast_like(qmlmath.eye(padding_left, like=interface), 1j), base_matrix)
        right_pad = qmlmath.convert_like(qmlmath.cast_like(qmlmath.eye(padding_right, like=interface), 1j), base_matrix)
        shape = qml.math.shape(base_matrix)
        if len(shape) == 3:
            return qml.math.stack([qml.math.block_diag([left_pad, _U, right_pad]) for _U in base_matrix])
        return qmlmath.block_diag([left_pad, base_matrix, right_pad])

    def matrix(self, wire_order=None):
        if self.compute_matrix is not Operator.compute_matrix:
            canonical_matrix = self.compute_matrix(*self.data)
        else:
            canonical_matrix = self._compute_matrix_from_base()
        wire_order = wire_order or self.wires
        return qml.math.expand_matrix(canonical_matrix, wires=self.active_wires, wire_order=wire_order)

    def sparse_matrix(self, wire_order=None, format='csr'):
        if wire_order is not None:
            raise NotImplementedError('wire_order argument is not yet implemented.')
        try:
            target_mat = self.base.sparse_matrix()
        except operation.SparseMatrixUndefinedError as e:
            if self.base.has_matrix:
                target_mat = sparse.lil_matrix(self.base.matrix())
            else:
                raise operation.SparseMatrixUndefinedError from e
        num_target_states = 2 ** len(self.target_wires)
        num_control_states = 2 ** len(self.control_wires)
        total_states = num_target_states * num_control_states
        start_ind = self._control_int * num_target_states
        end_ind = start_ind + num_target_states
        m = sparse.eye(total_states, format='lil', dtype=target_mat.dtype)
        m[start_ind:end_ind, start_ind:end_ind] = target_mat
        return m.asformat(format=format)

    def eigvals(self):
        base_eigvals = self.base.eigvals()
        num_target_wires = len(self.target_wires)
        num_control_wires = len(self.control_wires)
        total = 2 ** (num_target_wires + num_control_wires)
        ones = np.ones(total - len(base_eigvals))
        return qmlmath.concatenate([ones, base_eigvals])

    @property
    def has_diagonalizing_gates(self):
        return self.base.has_diagonalizing_gates

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    @property
    def has_decomposition(self):
        if self.compute_decomposition is not Operator.compute_decomposition:
            return True
        if not all(self.control_values):
            return True
        if len(self.control_wires) == 1 and hasattr(self.base, '_controlled'):
            return True
        if _is_single_qubit_special_unitary(self.base):
            return True
        if self.base.has_decomposition:
            return True
        return False

    def decomposition(self):
        if self.compute_decomposition is not Operator.compute_decomposition:
            return self.compute_decomposition(*self.data, self.wires)
        if all(self.control_values):
            decomp = _decompose_no_control_values(self)
            if decomp is None:
                raise qml.operation.DecompositionUndefinedError
            return decomp
        d = [qml.X(w) for w, val in zip(self.control_wires, self.control_values) if not val]
        decomp = _decompose_no_control_values(self)
        if decomp is None:
            no_control_values = copy(self).queue()
            no_control_values.hyperparameters['control_values'] = [1] * len(self.control_wires)
            d.append(no_control_values)
        else:
            d += decomp
        d += [qml.X(w) for w, val in zip(self.control_wires, self.control_values) if not val]
        return d

    @property
    def has_generator(self):
        return self.base.has_generator

    def generator(self):
        sub_gen = self.base.generator()
        projectors = (qml.Projector([val], wires=w) for val, w in zip(self.control_values, self.control_wires))
        return qml.prod(*projectors, sub_gen)

    @property
    def has_adjoint(self):
        return self.base.has_adjoint

    def adjoint(self):
        return ctrl(self.base.adjoint(), self.control_wires, control_values=self.control_values, work_wires=self.work_wires)

    def pow(self, z):
        base_pow = self.base.pow(z)
        return [ctrl(op, self.control_wires, control_values=self.control_values, work_wires=self.work_wires) for op in base_pow]

    def simplify(self) -> 'Operator':
        if isinstance(self.base, Controlled):
            base = self.base.base.simplify()
            return ctrl(base, control=self.control_wires + self.base.control_wires, control_values=self.control_values + self.base.control_values, work_wires=self.work_wires + self.base.work_wires)
        simplified_base = self.base.simplify()
        if isinstance(simplified_base, qml.Identity):
            return simplified_base
        return ctrl(op=simplified_base, control=self.control_wires, control_values=self.control_values, work_wires=self.work_wires)