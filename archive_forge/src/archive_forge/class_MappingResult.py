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
class MappingResult(_WithKeys, FilterResult[RowMapping]):
    """A wrapper for a :class:`_engine.Result` that returns dictionary values
    rather than :class:`_engine.Row` values.

    The :class:`_engine.MappingResult` object is acquired by calling the
    :meth:`_engine.Result.mappings` method.

    """
    __slots__ = ()
    _generate_rows = True
    _post_creational_filter = operator.attrgetter('_mapping')

    def __init__(self, result: Result[Any]):
        self._real_result = result
        self._unique_filter_state = result._unique_filter_state
        self._metadata = result._metadata
        if result._source_supports_scalars:
            self._metadata = self._metadata._reduce([0])

    def unique(self, strategy: Optional[_UniqueFilterType]=None) -> Self:
        """Apply unique filtering to the objects returned by this
        :class:`_engine.MappingResult`.

        See :meth:`_engine.Result.unique` for usage details.

        """
        self._unique_filter_state = (set(), strategy)
        return self

    def columns(self, *col_expressions: _KeyIndexType) -> Self:
        """Establish the columns that should be returned in each row."""
        return self._column_slices(col_expressions)

    def partitions(self, size: Optional[int]=None) -> Iterator[Sequence[RowMapping]]:
        """Iterate through sub-lists of elements of the size given.

        Equivalent to :meth:`_engine.Result.partitions` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        getter = self._manyrow_getter
        while True:
            partition = getter(self, size)
            if partition:
                yield partition
            else:
                break

    def fetchall(self) -> Sequence[RowMapping]:
        """A synonym for the :meth:`_engine.MappingResult.all` method."""
        return self._allrows()

    def fetchone(self) -> Optional[RowMapping]:
        """Fetch one object.

        Equivalent to :meth:`_engine.Result.fetchone` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        row = self._onerow_getter(self)
        if row is _NO_ROW:
            return None
        else:
            return row

    def fetchmany(self, size: Optional[int]=None) -> Sequence[RowMapping]:
        """Fetch many objects.

        Equivalent to :meth:`_engine.Result.fetchmany` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        return self._manyrow_getter(self, size)

    def all(self) -> Sequence[RowMapping]:
        """Return all scalar values in a sequence.

        Equivalent to :meth:`_engine.Result.all` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        return self._allrows()

    def __iter__(self) -> Iterator[RowMapping]:
        return self._iter_impl()

    def __next__(self) -> RowMapping:
        return self._next_impl()

    def first(self) -> Optional[RowMapping]:
        """Fetch the first object or ``None`` if no object is present.

        Equivalent to :meth:`_engine.Result.first` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.


        """
        return self._only_one_row(raise_for_second_row=False, raise_for_none=False, scalar=False)

    def one_or_none(self) -> Optional[RowMapping]:
        """Return at most one object or raise an exception.

        Equivalent to :meth:`_engine.Result.one_or_none` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        return self._only_one_row(raise_for_second_row=True, raise_for_none=False, scalar=False)

    def one(self) -> RowMapping:
        """Return exactly one object or raise an exception.

        Equivalent to :meth:`_engine.Result.one` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        return self._only_one_row(raise_for_second_row=True, raise_for_none=True, scalar=False)