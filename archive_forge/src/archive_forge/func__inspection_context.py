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
@contextlib.contextmanager
def _inspection_context(self) -> Generator[Inspector, None, None]:
    """Return an :class:`_reflection.Inspector`
        from this one that will run all
        operations on a single connection.

        """
    with self._operation_context() as conn:
        sub_insp = self._construct(self.__class__._init_connection, conn)
        sub_insp.info_cache = self.info_cache
        yield sub_insp