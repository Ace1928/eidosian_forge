from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple as typing_Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from .base import _NoArg
from .coercions import _document_text_coercion
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import Case
from .elements import Cast
from .elements import CollationClause
from .elements import CollectionAggregate
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Extract
from .elements import False_
from .elements import FunctionFilter
from .elements import Label
from .elements import Null
from .elements import Over
from .elements import TextClause
from .elements import True_
from .elements import TryCast
from .elements import Tuple
from .elements import TypeCoerce
from .elements import UnaryExpression
from .elements import WithinGroup
from .functions import FunctionElement
from ..util.typing import Literal
def type_coerce(expression: _ColumnExpressionOrLiteralArgument[Any], type_: _TypeEngineArgument[_T]) -> TypeCoerce[_T]:
    """Associate a SQL expression with a particular type, without rendering
    ``CAST``.

    E.g.::

        from sqlalchemy import type_coerce

        stmt = select(type_coerce(log_table.date_string, StringDateTime()))

    The above construct will produce a :class:`.TypeCoerce` object, which
    does not modify the rendering in any way on the SQL side, with the
    possible exception of a generated label if used in a columns clause
    context:

    .. sourcecode:: sql

        SELECT date_string AS date_string FROM log

    When result rows are fetched, the ``StringDateTime`` type processor
    will be applied to result rows on behalf of the ``date_string`` column.

    .. note:: the :func:`.type_coerce` construct does not render any
       SQL syntax of its own, including that it does not imply
       parenthesization.   Please use :meth:`.TypeCoerce.self_group`
       if explicit parenthesization is required.

    In order to provide a named label for the expression, use
    :meth:`_expression.ColumnElement.label`::

        stmt = select(
            type_coerce(log_table.date_string, StringDateTime()).label('date')
        )


    A type that features bound-value handling will also have that behavior
    take effect when literal values or :func:`.bindparam` constructs are
    passed to :func:`.type_coerce` as targets.
    For example, if a type implements the
    :meth:`.TypeEngine.bind_expression`
    method or :meth:`.TypeEngine.bind_processor` method or equivalent,
    these functions will take effect at statement compilation/execution
    time when a literal value is passed, as in::

        # bound-value handling of MyStringType will be applied to the
        # literal value "some string"
        stmt = select(type_coerce("some string", MyStringType))

    When using :func:`.type_coerce` with composed expressions, note that
    **parenthesis are not applied**.   If :func:`.type_coerce` is being
    used in an operator context where the parenthesis normally present from
    CAST are necessary, use the :meth:`.TypeCoerce.self_group` method:

    .. sourcecode:: pycon+sql

        >>> some_integer = column("someint", Integer)
        >>> some_string = column("somestr", String)
        >>> expr = type_coerce(some_integer + 5, String) + some_string
        >>> print(expr)
        {printsql}someint + :someint_1 || somestr{stop}
        >>> expr = type_coerce(some_integer + 5, String).self_group() + some_string
        >>> print(expr)
        {printsql}(someint + :someint_1) || somestr{stop}

    :param expression: A SQL expression, such as a
     :class:`_expression.ColumnElement`
     expression or a Python string which will be coerced into a bound
     literal value.

    :param type\\_: A :class:`.TypeEngine` class or instance indicating
     the type to which the expression is coerced.

    .. seealso::

        :ref:`tutorial_casts`

        :func:`.cast`

    """
    return TypeCoerce(expression, type_)