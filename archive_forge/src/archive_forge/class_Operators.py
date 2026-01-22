from __future__ import annotations
from enum import IntEnum
from operator import add as _uncast_add
from operator import and_ as _uncast_and_
from operator import contains as _uncast_contains
from operator import eq as _uncast_eq
from operator import floordiv as _uncast_floordiv
from operator import ge as _uncast_ge
from operator import getitem as _uncast_getitem
from operator import gt as _uncast_gt
from operator import inv as _uncast_inv
from operator import le as _uncast_le
from operator import lshift as _uncast_lshift
from operator import lt as _uncast_lt
from operator import mod as _uncast_mod
from operator import mul as _uncast_mul
from operator import ne as _uncast_ne
from operator import neg as _uncast_neg
from operator import or_ as _uncast_or_
from operator import rshift as _uncast_rshift
from operator import sub as _uncast_sub
from operator import truediv as _uncast_truediv
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import exc
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
class Operators:
    """Base of comparison and logical operators.

    Implements base methods
    :meth:`~sqlalchemy.sql.operators.Operators.operate` and
    :meth:`~sqlalchemy.sql.operators.Operators.reverse_operate`, as well as
    :meth:`~sqlalchemy.sql.operators.Operators.__and__`,
    :meth:`~sqlalchemy.sql.operators.Operators.__or__`,
    :meth:`~sqlalchemy.sql.operators.Operators.__invert__`.

    Usually is used via its most common subclass
    :class:`.ColumnOperators`.

    """
    __slots__ = ()

    def __and__(self, other: Any) -> Operators:
        """Implement the ``&`` operator.

        When used with SQL expressions, results in an
        AND operation, equivalent to
        :func:`_expression.and_`, that is::

            a & b

        is equivalent to::

            from sqlalchemy import and_
            and_(a, b)

        Care should be taken when using ``&`` regarding
        operator precedence; the ``&`` operator has the highest precedence.
        The operands should be enclosed in parenthesis if they contain
        further sub expressions::

            (a == 2) & (b == 4)

        """
        return self.operate(and_, other)

    def __or__(self, other: Any) -> Operators:
        """Implement the ``|`` operator.

        When used with SQL expressions, results in an
        OR operation, equivalent to
        :func:`_expression.or_`, that is::

            a | b

        is equivalent to::

            from sqlalchemy import or_
            or_(a, b)

        Care should be taken when using ``|`` regarding
        operator precedence; the ``|`` operator has the highest precedence.
        The operands should be enclosed in parenthesis if they contain
        further sub expressions::

            (a == 2) | (b == 4)

        """
        return self.operate(or_, other)

    def __invert__(self) -> Operators:
        """Implement the ``~`` operator.

        When used with SQL expressions, results in a
        NOT operation, equivalent to
        :func:`_expression.not_`, that is::

            ~a

        is equivalent to::

            from sqlalchemy import not_
            not_(a)

        """
        return self.operate(inv)

    def op(self, opstring: str, precedence: int=0, is_comparison: bool=False, return_type: Optional[Union[Type[TypeEngine[Any]], TypeEngine[Any]]]=None, python_impl: Optional[Callable[..., Any]]=None) -> Callable[[Any], Operators]:
        """Produce a generic operator function.

        e.g.::

          somecolumn.op("*")(5)

        produces::

          somecolumn * 5

        This function can also be used to make bitwise operators explicit. For
        example::

          somecolumn.op('&')(0xff)

        is a bitwise AND of the value in ``somecolumn``.

        :param opstring: a string which will be output as the infix operator
          between this element and the expression passed to the
          generated function.

        :param precedence: precedence which the database is expected to apply
         to the operator in SQL expressions. This integer value acts as a hint
         for the SQL compiler to know when explicit parenthesis should be
         rendered around a particular operation. A lower number will cause the
         expression to be parenthesized when applied against another operator
         with higher precedence. The default value of ``0`` is lower than all
         operators except for the comma (``,``) and ``AS`` operators. A value
         of 100 will be higher or equal to all operators, and -100 will be
         lower than or equal to all operators.

         .. seealso::

            :ref:`faq_sql_expression_op_parenthesis` - detailed description
            of how the SQLAlchemy SQL compiler renders parenthesis

        :param is_comparison: legacy; if True, the operator will be considered
         as a "comparison" operator, that is which evaluates to a boolean
         true/false value, like ``==``, ``>``, etc.  This flag is provided
         so that ORM relationships can establish that the operator is a
         comparison operator when used in a custom join condition.

         Using the ``is_comparison`` parameter is superseded by using the
         :meth:`.Operators.bool_op` method instead;  this more succinct
         operator sets this parameter automatically, but also provides
         correct :pep:`484` typing support as the returned object will
         express a "boolean" datatype, i.e. ``BinaryExpression[bool]``.

        :param return_type: a :class:`.TypeEngine` class or object that will
          force the return type of an expression produced by this operator
          to be of that type.   By default, operators that specify
          :paramref:`.Operators.op.is_comparison` will resolve to
          :class:`.Boolean`, and those that do not will be of the same
          type as the left-hand operand.

        :param python_impl: an optional Python function that can evaluate
         two Python values in the same way as this operator works when
         run on the database server.  Useful for in-Python SQL expression
         evaluation functions, such as for ORM hybrid attributes, and the
         ORM "evaluator" used to match objects in a session after a multi-row
         update or delete.

         e.g.::

            >>> expr = column('x').op('+', python_impl=lambda a, b: a + b)('y')

         The operator for the above expression will also work for non-SQL
         left and right objects::

            >>> expr.operator(5, 10)
            15

         .. versionadded:: 2.0


        .. seealso::

            :meth:`.Operators.bool_op`

            :ref:`types_operators`

            :ref:`relationship_custom_operator`

        """
        operator = custom_op(opstring, precedence, is_comparison, return_type, python_impl=python_impl)

        def against(other: Any) -> Operators:
            return operator(self, other)
        return against

    def bool_op(self, opstring: str, precedence: int=0, python_impl: Optional[Callable[..., Any]]=None) -> Callable[[Any], Operators]:
        """Return a custom boolean operator.

        This method is shorthand for calling
        :meth:`.Operators.op` and passing the
        :paramref:`.Operators.op.is_comparison`
        flag with True.    A key advantage to using :meth:`.Operators.bool_op`
        is that when using column constructs, the "boolean" nature of the
        returned expression will be present for :pep:`484` purposes.

        .. seealso::

            :meth:`.Operators.op`

        """
        return self.op(opstring, precedence=precedence, is_comparison=True, python_impl=python_impl)

    def operate(self, op: OperatorType, *other: Any, **kwargs: Any) -> Operators:
        """Operate on an argument.

        This is the lowest level of operation, raises
        :class:`NotImplementedError` by default.

        Overriding this on a subclass can allow common
        behavior to be applied to all operations.
        For example, overriding :class:`.ColumnOperators`
        to apply ``func.lower()`` to the left and right
        side::

            class MyComparator(ColumnOperators):
                def operate(self, op, other, **kwargs):
                    return op(func.lower(self), func.lower(other), **kwargs)

        :param op:  Operator callable.
        :param \\*other: the 'other' side of the operation. Will
         be a single scalar for most operations.
        :param \\**kwargs: modifiers.  These may be passed by special
         operators such as :meth:`ColumnOperators.contains`.


        """
        raise NotImplementedError(str(op))
    __sa_operate__ = operate

    def reverse_operate(self, op: OperatorType, other: Any, **kwargs: Any) -> Operators:
        """Reverse operate on an argument.

        Usage is the same as :meth:`operate`.

        """
        raise NotImplementedError(str(op))