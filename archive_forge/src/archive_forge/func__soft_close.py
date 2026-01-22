from __future__ import annotations
import collections
import functools
import operator
import typing
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .result import IteratorResult
from .result import MergedResult
from .result import Result
from .result import ResultMetaData
from .result import SimpleResultMetaData
from .result import tuplegetter
from .row import Row
from .. import exc
from .. import util
from ..sql import elements
from ..sql import sqltypes
from ..sql import util as sql_util
from ..sql.base import _generative
from ..sql.compiler import ResultColumnsEntry
from ..sql.compiler import RM_NAME
from ..sql.compiler import RM_OBJECTS
from ..sql.compiler import RM_RENDERED_NAME
from ..sql.compiler import RM_TYPE
from ..sql.type_api import TypeEngine
from ..util import compat
from ..util.typing import Literal
from ..util.typing import Self
def _soft_close(self, hard=False):
    """Soft close this :class:`_engine.CursorResult`.

        This releases all DBAPI cursor resources, but leaves the
        CursorResult "open" from a semantic perspective, meaning the
        fetchXXX() methods will continue to return empty results.

        This method is called automatically when:

        * all result rows are exhausted using the fetchXXX() methods.
        * cursor.description is None.

        This method is **not public**, but is documented in order to clarify
        the "autoclose" process used.

        .. seealso::

            :meth:`_engine.CursorResult.close`


        """
    if not hard and self._soft_closed or (hard and self.closed):
        return
    if hard:
        self.closed = True
        self.cursor_strategy.hard_close(self, self.cursor)
    else:
        self.cursor_strategy.soft_close(self, self.cursor)
    if not self._soft_closed:
        cursor = self.cursor
        self.cursor = None
        self.connection._safe_close_cursor(cursor)
        self._soft_closed = True