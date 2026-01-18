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
def tuple_(*clauses: _ColumnExpressionArgument[Any], types: Optional[Sequence[_TypeEngineArgument[Any]]]=None) -> Tuple:
    """Return a :class:`.Tuple`.

    Main usage is to produce a composite IN construct using
    :meth:`.ColumnOperators.in_` ::

        from sqlalchemy import tuple_

        tuple_(table.c.col1, table.c.col2).in_(
            [(1, 2), (5, 12), (10, 19)]
        )

    .. versionchanged:: 1.3.6 Added support for SQLite IN tuples.

    .. warning::

        The composite IN construct is not supported by all backends, and is
        currently known to work on PostgreSQL, MySQL, and SQLite.
        Unsupported backends will raise a subclass of
        :class:`~sqlalchemy.exc.DBAPIError` when such an expression is
        invoked.

    """
    return Tuple(*clauses, types=types)