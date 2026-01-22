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
class ScalarResult(FilterResult[_R]):
    """A wrapper for a :class:`_engine.Result` that returns scalar values
    rather than :class:`_row.Row` values.

    The :class:`_engine.ScalarResult` object is acquired by calling the
    :meth:`_engine.Result.scalars` method.

    A special limitation of :class:`_engine.ScalarResult` is that it has
    no ``fetchone()`` method; since the semantics of ``fetchone()`` are that
    the ``None`` value indicates no more results, this is not compatible
    with :class:`_engine.ScalarResult` since there is no way to distinguish
    between ``None`` as a row value versus ``None`` as an indicator.  Use
    ``next(result)`` to receive values individually.

    """
    __slots__ = ()
    _generate_rows = False
    _post_creational_filter: Optional[Callable[[Any], Any]]

    def __init__(self, real_result: Result[Any], index: _KeyIndexType):
        self._real_result = real_result
        if real_result._source_supports_scalars:
            self._metadata = real_result._metadata
            self._post_creational_filter = None
        else:
            self._metadata = real_result._metadata._reduce([index])
            self._post_creational_filter = operator.itemgetter(0)
        self._unique_filter_state = real_result._unique_filter_state

    def unique(self, strategy: Optional[_UniqueFilterType]=None) -> Self:
        """Apply unique filtering to the objects returned by this
        :class:`_engine.ScalarResult`.

        See :meth:`_engine.Result.unique` for usage details.

        """
        self._unique_filter_state = (set(), strategy)
        return self

    def partitions(self, size: Optional[int]=None) -> Iterator[Sequence[_R]]:
        """Iterate through sub-lists of elements of the size given.

        Equivalent to :meth:`_engine.Result.partitions` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        getter = self._manyrow_getter
        while True:
            partition = getter(self, size)
            if partition:
                yield partition
            else:
                break

    def fetchall(self) -> Sequence[_R]:
        """A synonym for the :meth:`_engine.ScalarResult.all` method."""
        return self._allrows()

    def fetchmany(self, size: Optional[int]=None) -> Sequence[_R]:
        """Fetch many objects.

        Equivalent to :meth:`_engine.Result.fetchmany` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        return self._manyrow_getter(self, size)

    def all(self) -> Sequence[_R]:
        """Return all scalar values in a sequence.

        Equivalent to :meth:`_engine.Result.all` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        return self._allrows()

    def __iter__(self) -> Iterator[_R]:
        return self._iter_impl()

    def __next__(self) -> _R:
        return self._next_impl()

    def first(self) -> Optional[_R]:
        """Fetch the first object or ``None`` if no object is present.

        Equivalent to :meth:`_engine.Result.first` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.


        """
        return self._only_one_row(raise_for_second_row=False, raise_for_none=False, scalar=False)

    def one_or_none(self) -> Optional[_R]:
        """Return at most one object or raise an exception.

        Equivalent to :meth:`_engine.Result.one_or_none` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        return self._only_one_row(raise_for_second_row=True, raise_for_none=False, scalar=False)

    def one(self) -> _R:
        """Return exactly one object or raise an exception.

        Equivalent to :meth:`_engine.Result.one` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        return self._only_one_row(raise_for_second_row=True, raise_for_none=True, scalar=False)