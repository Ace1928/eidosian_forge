from __future__ import annotations
from abc import ABC
import collections
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence as _typing_Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import ddl
from . import roles
from . import type_api
from . import visitors
from .base import _DefaultDescriptionTuple
from .base import _NoneName
from .base import _SentinelColumnCharacterization
from .base import _SentinelDefaultCharacterization
from .base import DedupeColumnCollection
from .base import DialectKWArgs
from .base import Executable
from .base import SchemaEventTarget as SchemaEventTarget
from .coercions import _document_text_coercion
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import quoted_name
from .elements import TextClause
from .selectable import TableClause
from .type_api import to_instance
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
class ColumnDefault(DefaultGenerator, ABC):
    """A plain default value on a column.

    This could correspond to a constant, a callable function,
    or a SQL clause.

    :class:`.ColumnDefault` is generated automatically
    whenever the ``default``, ``onupdate`` arguments of
    :class:`_schema.Column` are used.  A :class:`.ColumnDefault`
    can be passed positionally as well.

    For example, the following::

        Column('foo', Integer, default=50)

    Is equivalent to::

        Column('foo', Integer, ColumnDefault(50))


    """
    arg: Any

    @overload
    def __new__(cls, arg: Callable[..., Any], for_update: bool=...) -> CallableColumnDefault:
        ...

    @overload
    def __new__(cls, arg: ColumnElement[Any], for_update: bool=...) -> ColumnElementColumnDefault:
        ...

    @overload
    def __new__(cls, arg: object, for_update: bool=...) -> ColumnDefault:
        ...

    def __new__(cls, arg: Any=None, for_update: bool=False) -> ColumnDefault:
        """Construct a new :class:`.ColumnDefault`.


        :param arg: argument representing the default value.
         May be one of the following:

         * a plain non-callable Python value, such as a
           string, integer, boolean, or other simple type.
           The default value will be used as is each time.
         * a SQL expression, that is one which derives from
           :class:`_expression.ColumnElement`.  The SQL expression will
           be rendered into the INSERT or UPDATE statement,
           or in the case of a primary key column when
           RETURNING is not used may be
           pre-executed before an INSERT within a SELECT.
         * A Python callable.  The function will be invoked for each
           new row subject to an INSERT or UPDATE.
           The callable must accept exactly
           zero or one positional arguments.  The one-argument form
           will receive an instance of the :class:`.ExecutionContext`,
           which provides contextual information as to the current
           :class:`_engine.Connection` in use as well as the current
           statement and parameters.

        """
        if isinstance(arg, FetchedValue):
            raise exc.ArgumentError('ColumnDefault may not be a server-side default type.')
        elif callable(arg):
            cls = CallableColumnDefault
        elif isinstance(arg, ClauseElement):
            cls = ColumnElementColumnDefault
        elif arg is not None:
            cls = ScalarElementColumnDefault
        return object.__new__(cls)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.arg!r})'