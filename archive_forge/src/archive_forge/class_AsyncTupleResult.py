from __future__ import annotations
import operator
from typing import Any
from typing import AsyncIterator
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from . import exc as async_exc
from ... import util
from ...engine import Result
from ...engine.result import _NO_ROW
from ...engine.result import _R
from ...engine.result import _WithKeys
from ...engine.result import FilterResult
from ...engine.result import FrozenResult
from ...engine.result import ResultMetaData
from ...engine.row import Row
from ...engine.row import RowMapping
from ...sql.base import _generative
from ...util.concurrency import greenlet_spawn
from ...util.typing import Literal
from ...util.typing import Self
class AsyncTupleResult(AsyncCommon[_R], util.TypingOnly):
    """A :class:`_asyncio.AsyncResult` that's typed as returning plain
    Python tuples instead of rows.

    Since :class:`_engine.Row` acts like a tuple in every way already,
    this class is a typing only class, regular :class:`_asyncio.AsyncResult` is
    still used at runtime.

    """
    __slots__ = ()
    if TYPE_CHECKING:

        async def partitions(self, size: Optional[int]=None) -> AsyncIterator[Sequence[_R]]:
            """Iterate through sub-lists of elements of the size given.

            Equivalent to :meth:`_result.Result.partitions` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        async def fetchone(self) -> Optional[_R]:
            """Fetch one tuple.

            Equivalent to :meth:`_result.Result.fetchone` except that
            tuple values, rather than :class:`_engine.Row`
            objects, are returned.

            """
            ...

        async def fetchall(self) -> Sequence[_R]:
            """A synonym for the :meth:`_engine.ScalarResult.all` method."""
            ...

        async def fetchmany(self, size: Optional[int]=None) -> Sequence[_R]:
            """Fetch many objects.

            Equivalent to :meth:`_result.Result.fetchmany` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        async def all(self) -> Sequence[_R]:
            """Return all scalar values in a list.

            Equivalent to :meth:`_result.Result.all` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        async def __aiter__(self) -> AsyncIterator[_R]:
            ...

        async def __anext__(self) -> _R:
            ...

        async def first(self) -> Optional[_R]:
            """Fetch the first object or ``None`` if no object is present.

            Equivalent to :meth:`_result.Result.first` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.


            """
            ...

        async def one_or_none(self) -> Optional[_R]:
            """Return at most one object or raise an exception.

            Equivalent to :meth:`_result.Result.one_or_none` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        async def one(self) -> _R:
            """Return exactly one object or raise an exception.

            Equivalent to :meth:`_result.Result.one` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        @overload
        async def scalar_one(self: AsyncTupleResult[Tuple[_T]]) -> _T:
            ...

        @overload
        async def scalar_one(self) -> Any:
            ...

        async def scalar_one(self) -> Any:
            """Return exactly one scalar result or raise an exception.

            This is equivalent to calling :meth:`_engine.Result.scalars`
            and then :meth:`_engine.Result.one`.

            .. seealso::

                :meth:`_engine.Result.one`

                :meth:`_engine.Result.scalars`

            """
            ...

        @overload
        async def scalar_one_or_none(self: AsyncTupleResult[Tuple[_T]]) -> Optional[_T]:
            ...

        @overload
        async def scalar_one_or_none(self) -> Optional[Any]:
            ...

        async def scalar_one_or_none(self) -> Optional[Any]:
            """Return exactly one or no scalar result.

            This is equivalent to calling :meth:`_engine.Result.scalars`
            and then :meth:`_engine.Result.one_or_none`.

            .. seealso::

                :meth:`_engine.Result.one_or_none`

                :meth:`_engine.Result.scalars`

            """
            ...

        @overload
        async def scalar(self: AsyncTupleResult[Tuple[_T]]) -> Optional[_T]:
            ...

        @overload
        async def scalar(self) -> Any:
            ...

        async def scalar(self) -> Any:
            """Fetch the first column of the first row, and close the result
            set.

            Returns ``None`` if there are no rows to fetch.

            No validation is performed to test if additional rows remain.

            After calling this method, the object is fully closed,
            e.g. the :meth:`_engine.CursorResult.close`
            method will have been called.

            :return: a Python scalar value , or ``None`` if no rows remain.

            """
            ...