from __future__ import annotations
import copy
from numbers import Number
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.mixins import generate_apidocs
class ScalarOp(LinearOp):
    """Scalar identity operator class.

    This is a symbolic representation of an scalar identity operator on
    multiple subsystems. It may be used to initialize a symbolic scalar
    multiplication of an identity and then be implicitly converted to other
    kinds of operator subclasses by using the :meth:`compose`, :meth:`dot`,
    :meth:`tensor`, :meth:`expand` methods.
    """

    def __init__(self, dims: int | tuple | None=None, coeff: Number=1):
        """Initialize an operator object.

        Args:
            dims (int or tuple): subsystem dimensions.
            coeff (Number): scalar coefficient for the identity
                            operator (Default: 1).

        Raises:
            QiskitError: If the optional coefficient is invalid.
        """
        if not isinstance(coeff, Number):
            raise QiskitError(f'coeff {coeff} must be a number.')
        self._coeff = coeff
        super().__init__(input_dims=dims, output_dims=dims)

    def __array__(self, dtype=None):
        if dtype:
            return np.asarray(self.to_matrix(), dtype=dtype)
        return self.to_matrix()

    def __repr__(self):
        return f'ScalarOp({self.input_dims()}, coeff={self.coeff})'

    @property
    def coeff(self):
        """Return the coefficient"""
        return self._coeff

    def conjugate(self):
        ret = self.copy()
        ret._coeff = np.conjugate(self.coeff)
        return ret

    def transpose(self):
        return self.copy()

    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix."""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        return np.isclose(np.abs(self.coeff), 1, atol=atol, rtol=rtol)

    def to_matrix(self):
        """Convert to a Numpy matrix."""
        dim, _ = self.dim
        iden = np.eye(dim, dtype=complex)
        return self.coeff * iden

    def to_operator(self) -> Operator:
        """Convert to an Operator object."""
        return Operator(self.to_matrix(), input_dims=self.input_dims(), output_dims=self.output_dims())

    def compose(self, other: ScalarOp, qargs: list | None=None, front: bool=False) -> ScalarOp:
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if not isinstance(other, BaseOperator):
            other = Operator(other)
        new_shape = self._op_shape.compose(other._op_shape, qargs, front)
        if isinstance(other, ScalarOp):
            ret = copy.copy(self)
            ret._coeff = self.coeff * other.coeff
            ret._op_shape = new_shape
            return ret
        if qargs is None:
            ret = copy.copy(other)
            ret._op_shape = new_shape
            if self.coeff == 1:
                return ret
            return self.coeff * ret
        return other.__class__(self).compose(other, qargs=qargs, front=front)

    def power(self, n: float) -> ScalarOp:
        """Return the power of the ScalarOp.

        Args:
            n (float): the exponent for the scalar op.

        Returns:
            ScalarOp: the ``coeff ** n`` ScalarOp.
        """
        ret = self.copy()
        ret._coeff = self.coeff ** n
        return ret

    def tensor(self, other: ScalarOp) -> ScalarOp:
        if not isinstance(other, BaseOperator):
            other = Operator(other)
        if isinstance(other, ScalarOp):
            ret = copy.copy(self)
            ret._coeff = self.coeff * other.coeff
            ret._op_shape = self._op_shape.tensor(other._op_shape)
            return ret
        return other.expand(self)

    def expand(self, other: ScalarOp) -> ScalarOp:
        if not isinstance(other, BaseOperator):
            other = Operator(other)
        if isinstance(other, ScalarOp):
            ret = copy.copy(self)
            ret._coeff = self.coeff * other.coeff
            ret._op_shape = self._op_shape.expand(other._op_shape)
            return ret
        return other.tensor(self)

    def _add(self, other, qargs=None):
        """Return the operator self + other.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (BaseOperator): an operator object.
            qargs (None or list): optional subsystems to subtract on
                                  (Default: None)

        Returns:
            ScalarOp: if other is an ScalarOp.
            BaseOperator: if other is not an ScalarOp.

        Raises:
            QiskitError: if other has incompatible dimensions.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if not isinstance(other, BaseOperator):
            other = Operator(other)
        self._op_shape._validate_add(other._op_shape, qargs)
        if isinstance(other, ScalarOp):
            return ScalarOp(self.input_dims(), coeff=self.coeff + other.coeff)
        other = ScalarOp._pad_with_identity(self, other, qargs)
        if self.coeff == 0:
            return other.reshape(self.input_dims(), self.output_dims())
        return other.reshape(self.input_dims(), self.output_dims())._add(self)

    def _multiply(self, other):
        """Return the ScalarOp other * self.

        Args:
            other (Number): a complex number.

        Returns:
            ScalarOp: the scaled identity operator other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError(f'other ({other}) is not a number')
        ret = self.copy()
        ret._coeff = other * self.coeff
        return ret

    @staticmethod
    def _pad_with_identity(current, other, qargs=None):
        """Pad another operator with identities.

        Args:
            current (BaseOperator): current operator.
            other (BaseOperator): other operator.
            qargs (None or list): qargs

        Returns:
            BaseOperator: the padded operator.
        """
        if qargs is None:
            return other
        return ScalarOp(current.input_dims()).compose(other, qargs=qargs)