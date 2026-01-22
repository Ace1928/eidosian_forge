import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
class LinearBase(metaclass=abc.ABCMeta):
    """Interface for types that can build linear expressions with +, -, * and /.

    Classes derived from LinearBase (plus float and int scalars) are used to
    build expression trees describing a linear expression. Operations nodes of the
    expression tree include:

      * LinearSum: describes a deferred sum of LinearTypes objects.
      * LinearProduct: describes a deferred product of a scalar and a
        LinearTypes object.

    Leaf nodes of the expression tree include:

      * float and int scalars.
      * Variable: a single variable.
      * LinearTerm: the product of a scalar and a Variable object.
      * LinearExpression: the sum of a scalar and LinearTerm objects.

    LinearBase objects/expression-trees can be used directly to create
    constraints or objective functions. However, to facilitate their inspection,
    any LinearTypes object can be flattened to a LinearExpression
    through:

     as_flat_linear_expression(value: LinearTypes) -> LinearExpression:

    In addition, all LinearBase objects are immutable.

    Performance notes:

    Using an expression tree representation instead of an eager construction of
    LinearExpression objects reduces known inefficiencies associated with the
    use of operator overloading to construct linear expressions. In particular, we
    expect the runtime of as_flat_linear_expression() to be linear in the size of
    the expression tree. Additional performance can gained by using LinearSum(c)
    instead of sum(c) for a container c, as the latter creates len(c) LinearSum
    objects.
    """
    __slots__ = ()

    @abc.abstractmethod
    def _flatten_once_and_add_to(self, scale: float, processed_elements: _ProcessedElements, target_stack: _ToProcessElements) -> None:
        """Flatten one level of tree if needed and add to targets.

        Classes derived from LinearBase only need to implement this function
        to enable transformation to LinearExpression through
        as_flat_linear_expression().

        Args:
          scale: multiply elements by this number when processing or adding to
            stack.
          processed_elements: where to add LinearTerms and scalars that can be
            processed immediately.
          target_stack: where to add LinearBase elements that are not scalars or
            LinearTerms (i.e. elements that need further flattening).
            Implementations should append() to this stack to avoid being recursive.
        """

    def __eq__(self, rhs: LinearTypes) -> 'BoundedLinearExpression':
        if isinstance(rhs, (int, float)):
            return BoundedLinearExpression(rhs, self, rhs)
        if not isinstance(rhs, LinearBase):
            _raise_binary_operator_type_error('==', type(self), type(rhs))
        return BoundedLinearExpression(0.0, self - rhs, 0.0)

    def __ne__(self, rhs: LinearTypes) -> NoReturn:
        _raise_ne_not_supported()

    @typing.overload
    def __le__(self, rhs: float) -> 'UpperBoundedLinearExpression':
        ...

    @typing.overload
    def __le__(self, rhs: 'LinearBase') -> 'BoundedLinearExpression':
        ...

    @typing.overload
    def __le__(self, rhs: 'BoundedLinearExpression') -> NoReturn:
        ...

    def __le__(self, rhs):
        if isinstance(rhs, (int, float)):
            return UpperBoundedLinearExpression(self, rhs)
        if isinstance(rhs, LinearBase):
            return BoundedLinearExpression(-math.inf, self - rhs, 0.0)
        if isinstance(rhs, BoundedLinearExpression):
            _raise_binary_operator_type_error('<=', type(self), type(rhs), _EXPRESSION_COMP_EXPRESSION_MESSAGE)
        _raise_binary_operator_type_error('<=', type(self), type(rhs))

    @typing.overload
    def __ge__(self, lhs: float) -> 'LowerBoundedLinearExpression':
        ...

    @typing.overload
    def __ge__(self, lhs: 'LinearBase') -> 'BoundedLinearExpression':
        ...

    @typing.overload
    def __ge__(self, lhs: 'BoundedLinearExpression') -> NoReturn:
        ...

    def __ge__(self, lhs):
        if isinstance(lhs, (int, float)):
            return LowerBoundedLinearExpression(self, lhs)
        if isinstance(lhs, LinearBase):
            return BoundedLinearExpression(0.0, self - lhs, math.inf)
        if isinstance(lhs, BoundedLinearExpression):
            _raise_binary_operator_type_error('>=', type(self), type(lhs), _EXPRESSION_COMP_EXPRESSION_MESSAGE)
        _raise_binary_operator_type_error('>=', type(self), type(lhs))

    def __add__(self, expr: LinearTypes) -> 'LinearSum':
        if not isinstance(expr, (int, float, LinearBase)):
            return NotImplemented
        return LinearSum((self, expr))

    def __radd__(self, expr: LinearTypes) -> 'LinearSum':
        if not isinstance(expr, (int, float, LinearBase)):
            return NotImplemented
        return LinearSum((expr, self))

    def __sub__(self, expr: LinearTypes) -> 'LinearSum':
        if not isinstance(expr, (int, float, LinearBase)):
            return NotImplemented
        return LinearSum((self, -expr))

    def __rsub__(self, expr: LinearTypes) -> 'LinearSum':
        if not isinstance(expr, (int, float, LinearBase, QuadraticBase)):
            return NotImplemented
        return LinearSum((expr, -self))

    @typing.overload
    def __mul__(self, other: float) -> 'LinearProduct':
        ...

    @typing.overload
    def __mul__(self, other: 'LinearBase') -> 'LinearLinearProduct':
        ...

    def __mul__(self, other):
        if not isinstance(other, (int, float, LinearBase)):
            return NotImplemented
        if isinstance(other, LinearBase):
            return LinearLinearProduct(self, other)
        return LinearProduct(other, self)

    def __rmul__(self, constant: float) -> 'LinearProduct':
        if not isinstance(constant, (int, float)):
            return NotImplemented
        return LinearProduct(constant, self)

    def __truediv__(self, constant: float) -> 'LinearProduct':
        if not isinstance(constant, (int, float)):
            return NotImplemented
        return LinearProduct(1.0 / constant, self)

    def __neg__(self) -> 'LinearProduct':
        return LinearProduct(-1.0, self)