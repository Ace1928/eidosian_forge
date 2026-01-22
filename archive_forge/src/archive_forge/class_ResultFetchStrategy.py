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
class ResultFetchStrategy:
    """Define a fetching strategy for a result object.


    .. versionadded:: 1.4

    """
    __slots__ = ()
    alternate_cursor_description: Optional[_DBAPICursorDescription] = None

    def soft_close(self, result: CursorResult[Any], dbapi_cursor: Optional[DBAPICursor]) -> None:
        raise NotImplementedError()

    def hard_close(self, result: CursorResult[Any], dbapi_cursor: Optional[DBAPICursor]) -> None:
        raise NotImplementedError()

    def yield_per(self, result: CursorResult[Any], dbapi_cursor: Optional[DBAPICursor], num: int) -> None:
        return

    def fetchone(self, result: CursorResult[Any], dbapi_cursor: DBAPICursor, hard_close: bool=False) -> Any:
        raise NotImplementedError()

    def fetchmany(self, result: CursorResult[Any], dbapi_cursor: DBAPICursor, size: Optional[int]=None) -> Any:
        raise NotImplementedError()

    def fetchall(self, result: CursorResult[Any], dbapi_cursor: DBAPICursor) -> Any:
        raise NotImplementedError()

    def handle_exception(self, result: CursorResult[Any], dbapi_cursor: Optional[DBAPICursor], err: BaseException) -> NoReturn:
        raise err