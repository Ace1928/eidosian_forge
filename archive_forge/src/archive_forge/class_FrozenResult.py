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
class FrozenResult(Generic[_TP]):
    """Represents a :class:`_engine.Result` object in a "frozen" state suitable
    for caching.

    The :class:`_engine.FrozenResult` object is returned from the
    :meth:`_engine.Result.freeze` method of any :class:`_engine.Result`
    object.

    A new iterable :class:`_engine.Result` object is generated from a fixed
    set of data each time the :class:`_engine.FrozenResult` is invoked as
    a callable::


        result = connection.execute(query)

        frozen = result.freeze()

        unfrozen_result_one = frozen()

        for row in unfrozen_result_one:
            print(row)

        unfrozen_result_two = frozen()
        rows = unfrozen_result_two.all()

        # ... etc

    .. versionadded:: 1.4

    .. seealso::

        :ref:`do_orm_execute_re_executing` - example usage within the
        ORM to implement a result-set cache.

        :func:`_orm.loading.merge_frozen_result` - ORM function to merge
        a frozen result back into a :class:`_orm.Session`.

    """
    data: Sequence[Any]

    def __init__(self, result: Result[_TP]):
        self.metadata = result._metadata._for_freeze()
        self._source_supports_scalars = result._source_supports_scalars
        self._attributes = result._attributes
        if self._source_supports_scalars:
            self.data = list(result._raw_row_iterator())
        else:
            self.data = result.fetchall()

    def rewrite_rows(self) -> Sequence[Sequence[Any]]:
        if self._source_supports_scalars:
            return [[elem] for elem in self.data]
        else:
            return [list(row) for row in self.data]

    def with_new_rows(self, tuple_data: Sequence[Row[_TP]]) -> FrozenResult[_TP]:
        fr = FrozenResult.__new__(FrozenResult)
        fr.metadata = self.metadata
        fr._attributes = self._attributes
        fr._source_supports_scalars = self._source_supports_scalars
        if self._source_supports_scalars:
            fr.data = [d[0] for d in tuple_data]
        else:
            fr.data = tuple_data
        return fr

    def __call__(self) -> Result[_TP]:
        result: IteratorResult[_TP] = IteratorResult(self.metadata, iter(self.data))
        result._attributes = self._attributes
        result._source_supports_scalars = self._source_supports_scalars
        return result