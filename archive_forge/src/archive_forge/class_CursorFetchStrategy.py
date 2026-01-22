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
class CursorFetchStrategy(ResultFetchStrategy):
    """Call fetch methods from a DBAPI cursor.

    Alternate versions of this class may instead buffer the rows from
    cursors or not use cursors at all.

    """
    __slots__ = ()

    def soft_close(self, result: CursorResult[Any], dbapi_cursor: Optional[DBAPICursor]) -> None:
        result.cursor_strategy = _NO_CURSOR_DQL

    def hard_close(self, result: CursorResult[Any], dbapi_cursor: Optional[DBAPICursor]) -> None:
        result.cursor_strategy = _NO_CURSOR_DQL

    def handle_exception(self, result: CursorResult[Any], dbapi_cursor: Optional[DBAPICursor], err: BaseException) -> NoReturn:
        result.connection._handle_dbapi_exception(err, None, None, dbapi_cursor, result.context)

    def yield_per(self, result: CursorResult[Any], dbapi_cursor: Optional[DBAPICursor], num: int) -> None:
        result.cursor_strategy = BufferedRowCursorFetchStrategy(dbapi_cursor, {'max_row_buffer': num}, initial_buffer=collections.deque(), growth_factor=0)

    def fetchone(self, result: CursorResult[Any], dbapi_cursor: DBAPICursor, hard_close: bool=False) -> Any:
        try:
            row = dbapi_cursor.fetchone()
            if row is None:
                result._soft_close(hard=hard_close)
            return row
        except BaseException as e:
            self.handle_exception(result, dbapi_cursor, e)

    def fetchmany(self, result: CursorResult[Any], dbapi_cursor: DBAPICursor, size: Optional[int]=None) -> Any:
        try:
            if size is None:
                l = dbapi_cursor.fetchmany()
            else:
                l = dbapi_cursor.fetchmany(size)
            if not l:
                result._soft_close()
            return l
        except BaseException as e:
            self.handle_exception(result, dbapi_cursor, e)

    def fetchall(self, result: CursorResult[Any], dbapi_cursor: DBAPICursor) -> Any:
        try:
            rows = dbapi_cursor.fetchall()
            result._soft_close()
            return rows
        except BaseException as e:
            self.handle_exception(result, dbapi_cursor, e)