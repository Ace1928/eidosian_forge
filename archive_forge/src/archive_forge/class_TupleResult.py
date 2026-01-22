from __future__ import annotations
from enum import Enum
import functools
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .row import Row
from .row import RowMapping
from .. import exc
from .. import util
from ..sql.base import _generative
from ..sql.base import HasMemoized
from ..sql.base import InPlaceGenerative
from ..util import HasMemoized_ro_memoized_attribute
from ..util import NONE_SET
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Self
class TupleResult(FilterResult[_R], util.TypingOnly):
    """A :class:`_engine.Result` that's typed as returning plain
    Python tuples instead of rows.

    Since :class:`_engine.Row` acts like a tuple in every way already,
    this class is a typing only class, regular :class:`_engine.Result` is
    still used at runtime.

    """
    __slots__ = ()
    if TYPE_CHECKING:

        def partitions(self, size: Optional[int]=None) -> Iterator[Sequence[_R]]:
            """Iterate through sub-lists of elements of the size given.

            Equivalent to :meth:`_engine.Result.partitions` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        def fetchone(self) -> Optional[_R]:
            """Fetch one tuple.

            Equivalent to :meth:`_engine.Result.fetchone` except that
            tuple values, rather than :class:`_engine.Row`
            objects, are returned.

            """
            ...

        def fetchall(self) -> Sequence[_R]:
            """A synonym for the :meth:`_engine.ScalarResult.all` method."""
            ...

        def fetchmany(self, size: Optional[int]=None) -> Sequence[_R]:
            """Fetch many objects.

            Equivalent to :meth:`_engine.Result.fetchmany` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        def all(self) -> Sequence[_R]:
            """Return all scalar values in a sequence.

            Equivalent to :meth:`_engine.Result.all` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        def __iter__(self) -> Iterator[_R]:
            ...

        def __next__(self) -> _R:
            ...

        def first(self) -> Optional[_R]:
            """Fetch the first object or ``None`` if no object is present.

            Equivalent to :meth:`_engine.Result.first` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.


            """
            ...

        def one_or_none(self) -> Optional[_R]:
            """Return at most one object or raise an exception.

            Equivalent to :meth:`_engine.Result.one_or_none` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        def one(self) -> _R:
            """Return exactly one object or raise an exception.

            Equivalent to :meth:`_engine.Result.one` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        @overload
        def scalar_one(self: TupleResult[Tuple[_T]]) -> _T:
            ...

        @overload
        def scalar_one(self) -> Any:
            ...

        def scalar_one(self) -> Any:
            """Return exactly one scalar result or raise an exception.

            This is equivalent to calling :meth:`_engine.Result.scalars`
            and then :meth:`_engine.Result.one`.

            .. seealso::

                :meth:`_engine.Result.one`

                :meth:`_engine.Result.scalars`

            """
            ...

        @overload
        def scalar_one_or_none(self: TupleResult[Tuple[_T]]) -> Optional[_T]:
            ...

        @overload
        def scalar_one_or_none(self) -> Optional[Any]:
            ...

        def scalar_one_or_none(self) -> Optional[Any]:
            """Return exactly one or no scalar result.

            This is equivalent to calling :meth:`_engine.Result.scalars`
            and then :meth:`_engine.Result.one_or_none`.

            .. seealso::

                :meth:`_engine.Result.one_or_none`

                :meth:`_engine.Result.scalars`

            """
            ...

        @overload
        def scalar(self: TupleResult[Tuple[_T]]) -> Optional[_T]:
            ...

        @overload
        def scalar(self) -> Any:
            ...

        def scalar(self) -> Any:
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