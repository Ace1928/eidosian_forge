from __future__ import annotations
import collections.abc as collections_abc
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from . import util as sql_util
from ._typing import _TP
from ._typing import _unexpected_kw
from ._typing import is_column_element
from ._typing import is_named_from_clause
from .base import _entity_namespace_key
from .base import _exclusive_against
from .base import _from_objects
from .base import _generative
from .base import _select_iterables
from .base import ColumnCollection
from .base import CompileState
from .base import DialectKWArgs
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Null
from .selectable import Alias
from .selectable import ExecutableReturnsRows
from .selectable import FromClause
from .selectable import HasCTE
from .selectable import HasPrefixes
from .selectable import Join
from .selectable import SelectLabelStyle
from .selectable import TableClause
from .selectable import TypedReturnsRows
from .sqltypes import NullType
from .visitors import InternalTraversal
from .. import exc
from .. import util
from ..util.typing import Self
from ..util.typing import TypeGuard
@_generative
def with_dialect_options(self, **opt: Any) -> Self:
    """Add dialect options to this INSERT/UPDATE/DELETE object.

        e.g.::

            upd = table.update().dialect_options(mysql_limit=10)

        .. versionadded: 1.4 - this method supersedes the dialect options
           associated with the constructor.


        """
    self._validate_dialect_kwargs(opt)
    return self