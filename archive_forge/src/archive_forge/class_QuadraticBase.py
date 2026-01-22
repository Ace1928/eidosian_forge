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
class QuadraticBase(metaclass=abc.ABCMeta):
    """Interface for types that can build quadratic expressions with +, -, * and /.

    Classes derived from QuadraticBase and LinearBase (plus float and int scalars)
    are used to build expression trees describing a quadratic expression.
    Operations nodes of the expression tree include:

        * QuadraticSum: describes a deferred sum of QuadraticTypes objects.
        * QuadraticProduct: describes a deferred product of a scalar and a
          QuadraticTypes object.
        * LinearLinearProduct: describes a deferred product of two LinearTypes
          objects.

      Leaf nodes of the expression tree include:

        * float and int scalars.
        * Variable: a single variable.
        * LinearTerm: the product of a scalar and a Variable object.
        * LinearExpression: the sum of a scalar and LinearTerm objects.
        * QuadraticTerm: the product of a scalar and two Variable objects.
        * QuadraticExpression: the sum of a scalar, LinearTerm objects and
          QuadraticTerm objects.

    QuadraticBase objects/expression-trees can be used directly to create
    objective functions. However, to facilitate their inspection, any
    QuadraticTypes object can be flattened to a QuadraticExpression
    through:

     as_flat_quadratic_expression(value: QuadraticTypes) -> QuadraticExpression:

    In addition, all QuadraticBase objects are immutable.

    Performance notes:

    Using an expression tree representation instead of an eager construction of
    QuadraticExpression objects reduces known inefficiencies associated with the
    use of operator overloading to construct quadratic expressions. In particular,
    we expect the runtime of as_flat_quadratic_expression() to be linear in the
    size of the expression tree. Additional performance can gained by using
    QuadraticSum(c) instead of sum(c) for a container c, as the latter creates
    len(c) QuadraticSum objects.
    """
    __slots__ = ()

    @abc.abstractmethod
    def _quadratic_flatten_once_and_add_to(self, scale: float, processed_elements: _QuadraticProcessedElements, target_stack: _QuadraticToProcessElements) -> None:
        """Flatten one level of tree if needed and add to targets.

        Classes derived from QuadraticBase only need to implement this function
        to enable transformation to QuadraticExpression through
        as_flat_quadratic_expression().

        Args:
          scale: multiply elements by this number when processing or adding to
            stack.
          processed_elements: where to add linear terms, quadratic terms and scalars
            that can be processed immediately.
          target_stack: where to add LinearBase and QuadraticBase elements that are
            not scalars or linear terms or quadratic terms (i.e. elements that need
            further flattening). Implementations should append() to this stack to
            avoid being recursive.
        """

    def __add__(self, expr: QuadraticTypes) -> 'QuadraticSum':
        if not isinstance(expr, (int, float, LinearBase, QuadraticBase)):
            return NotImplemented
        return QuadraticSum((self, expr))

    def __radd__(self, expr: QuadraticTypes) -> 'QuadraticSum':
        if not isinstance(expr, (int, float, LinearBase, QuadraticBase)):
            return NotImplemented
        return QuadraticSum((expr, self))

    def __sub__(self, expr: QuadraticTypes) -> 'QuadraticSum':
        if not isinstance(expr, (int, float, LinearBase, QuadraticBase)):
            return NotImplemented
        return QuadraticSum((self, -expr))

    def __rsub__(self, expr: QuadraticTypes) -> 'QuadraticSum':
        if not isinstance(expr, (int, float, LinearBase, QuadraticBase)):
            return NotImplemented
        return QuadraticSum((expr, -self))

    def __mul__(self, other: float) -> 'QuadraticProduct':
        if not isinstance(other, (int, float)):
            return NotImplemented
        return QuadraticProduct(other, self)

    def __rmul__(self, other: float) -> 'QuadraticProduct':
        if not isinstance(other, (int, float)):
            return NotImplemented
        return QuadraticProduct(other, self)

    def __truediv__(self, constant: float) -> 'QuadraticProduct':
        if not isinstance(constant, (int, float)):
            return NotImplemented
        return QuadraticProduct(1.0 / constant, self)

    def __neg__(self) -> 'QuadraticProduct':
        return QuadraticProduct(-1.0, self)