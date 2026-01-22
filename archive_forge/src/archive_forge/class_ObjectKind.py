from __future__ import annotations
import contextlib
from dataclasses import dataclass
from enum import auto
from enum import Flag
from enum import unique
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import Connection
from .base import Engine
from .. import exc
from .. import inspection
from .. import sql
from .. import util
from ..sql import operators
from ..sql import schema as sa_schema
from ..sql.cache_key import _ad_hoc_cache_key_from_args
from ..sql.elements import TextClause
from ..sql.type_api import TypeEngine
from ..sql.visitors import InternalTraversal
from ..util import topological
from ..util.typing import final
@unique
class ObjectKind(Flag):
    """Enumerator that indicates which kind of object to return when calling
    the ``get_multi`` methods.

    This is a Flag enum, so custom combinations can be passed. For example,
    to reflect tables and plain views ``ObjectKind.TABLE | ObjectKind.VIEW``
    may be used.

    .. note::
      Not all dialect may support all kind of object. If a dialect does
      not support a particular object an empty dict is returned.
      In case a dialect supports an object, but the requested method
      is not applicable for the specified kind the default value
      will be returned for each reflected object. For example reflecting
      check constraints of view return a dict with all the views with
      empty lists as values.
    """
    TABLE = auto()
    'Reflect table objects'
    VIEW = auto()
    'Reflect plain view objects'
    MATERIALIZED_VIEW = auto()
    'Reflect materialized view object'
    ANY_VIEW = VIEW | MATERIALIZED_VIEW
    'Reflect any kind of view objects'
    ANY = TABLE | VIEW | MATERIALIZED_VIEW
    'Reflect all type of objects'