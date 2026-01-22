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
class DefaultGenerator(Executable, SchemaItem):
    """Base class for column *default* values.

    This object is only present on column.default or column.onupdate.
    It's not valid as a server default.

    """
    __visit_name__ = 'default_generator'
    _is_default_generator = True
    is_sequence = False
    is_identity = False
    is_server_default = False
    is_clause_element = False
    is_callable = False
    is_scalar = False
    has_arg = False
    is_sentinel = False
    column: Optional[Column[Any]]

    def __init__(self, for_update: bool=False) -> None:
        self.for_update = for_update

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        if TYPE_CHECKING:
            assert isinstance(parent, Column)
        self.column = parent
        if self.for_update:
            self.column.onupdate = self
        else:
            self.column.default = self

    def _copy(self) -> DefaultGenerator:
        raise NotImplementedError()

    def _execute_on_connection(self, connection: Connection, distilled_params: _CoreMultiExecuteParams, execution_options: CoreExecuteOptionsParameter) -> Any:
        util.warn_deprecated('Using the .execute() method to invoke a DefaultGenerator object is deprecated; please use the .scalar() method.', '2.0')
        return self._execute_on_scalar(connection, distilled_params, execution_options)

    def _execute_on_scalar(self, connection: Connection, distilled_params: _CoreMultiExecuteParams, execution_options: CoreExecuteOptionsParameter) -> Any:
        return connection._execute_default(self, distilled_params, execution_options)