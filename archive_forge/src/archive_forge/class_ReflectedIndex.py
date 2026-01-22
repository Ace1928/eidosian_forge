from __future__ import annotations
from enum import Enum
from types import ModuleType
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..event import EventTarget
from ..pool import Pool
from ..pool import PoolProxiedConnection
from ..sql.compiler import Compiled as Compiled
from ..sql.compiler import Compiled  # noqa
from ..sql.compiler import TypeCompiler as TypeCompiler
from ..sql.compiler import TypeCompiler  # noqa
from ..util import immutabledict
from ..util.concurrency import await_only
from ..util.typing import Literal
from ..util.typing import NotRequired
from ..util.typing import Protocol
from ..util.typing import TypedDict
class ReflectedIndex(TypedDict):
    """Dictionary representing the reflected elements corresponding to
    :class:`.Index`.

    The :class:`.ReflectedIndex` structure is returned by the
    :meth:`.Inspector.get_indexes` method.

    """
    name: Optional[str]
    'index name'
    column_names: List[Optional[str]]
    "column names which the index references.\n    An element of this list is ``None`` if it's an expression and is\n    returned in the ``expressions`` list.\n    "
    expressions: NotRequired[List[str]]
    'Expressions that compose the index. This list, when present, contains\n    both plain column names (that are also in ``column_names``) and\n    expressions (that are ``None`` in ``column_names``).\n    '
    unique: bool
    'whether or not the index has a unique flag'
    duplicates_constraint: NotRequired[Optional[str]]
    'Indicates if this index mirrors a constraint with this name'
    include_columns: NotRequired[List[str]]
    'columns to include in the INCLUDE clause for supporting databases.\n\n    .. deprecated:: 2.0\n\n        Legacy value, will be replaced with\n        ``index_dict["dialect_options"]["<dialect name>_include"]``\n\n    '
    column_sorting: NotRequired[Dict[str, Tuple[str]]]
    'optional dict mapping column names or expressions to tuple of sort\n    keywords, which may include ``asc``, ``desc``, ``nulls_first``,\n    ``nulls_last``.\n\n    .. versionadded:: 1.3.5\n    '
    dialect_options: NotRequired[Dict[str, Any]]
    'Additional dialect-specific options detected for this index'