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
class DefaultClause(FetchedValue):
    """A DDL-specified DEFAULT column value.

    :class:`.DefaultClause` is a :class:`.FetchedValue`
    that also generates a "DEFAULT" clause when
    "CREATE TABLE" is emitted.

    :class:`.DefaultClause` is generated automatically
    whenever the ``server_default``, ``server_onupdate`` arguments of
    :class:`_schema.Column` are used.  A :class:`.DefaultClause`
    can be passed positionally as well.

    For example, the following::

        Column('foo', Integer, server_default="50")

    Is equivalent to::

        Column('foo', Integer, DefaultClause("50"))

    """
    has_argument = True

    def __init__(self, arg: Union[str, ClauseElement, TextClause], for_update: bool=False, _reflected: bool=False) -> None:
        util.assert_arg_type(arg, (str, ClauseElement, TextClause), 'arg')
        super().__init__(for_update)
        self.arg = arg
        self.reflected = _reflected

    def _copy(self) -> DefaultClause:
        return DefaultClause(arg=self.arg, for_update=self.for_update, _reflected=self.reflected)

    def __repr__(self) -> str:
        return 'DefaultClause(%r, for_update=%r)' % (self.arg, self.for_update)