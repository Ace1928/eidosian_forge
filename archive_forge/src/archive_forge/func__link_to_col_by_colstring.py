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
def _link_to_col_by_colstring(self, parenttable: Table, table: Table, colname: Optional[str]) -> Column[Any]:
    _column = None
    if colname is None:
        parent = self.parent
        assert parent is not None
        key = parent.key
        _column = table.c.get(key, None)
    elif self.link_to_name:
        key = colname
        for c in table.c:
            if c.name == colname:
                _column = c
    else:
        key = colname
        _column = table.c.get(colname, None)
    if _column is None:
        raise exc.NoReferencedColumnError(f"Could not initialize target column for ForeignKey '{self._colspec}' on table '{parenttable.name}': table '{table.name}' has no column named '{key}'", table.name, key)
    return _column