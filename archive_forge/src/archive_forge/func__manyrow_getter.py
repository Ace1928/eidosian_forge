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
@HasMemoized_ro_memoized_attribute
def _manyrow_getter(self) -> Callable[..., List[_R]]:
    make_row = self._row_getter
    post_creational_filter = self._post_creational_filter
    if self._unique_filter_state:
        uniques, strategy = self._unique_strategy

        def filterrows(make_row: Optional[Callable[..., _R]], rows: List[Any], strategy: Optional[Callable[[List[Any]], Any]], uniques: Set[Any]) -> List[_R]:
            if make_row:
                rows = [make_row(row) for row in rows]
            if strategy:
                made_rows = ((made_row, strategy(made_row)) for made_row in rows)
            else:
                made_rows = ((made_row, made_row) for made_row in rows)
            return [made_row for made_row, sig_row in made_rows if sig_row not in uniques and (not uniques.add(sig_row))]

        def manyrows(self: ResultInternal[_R], num: Optional[int]) -> List[_R]:
            collect: List[_R] = []
            _manyrows = self._fetchmany_impl
            if num is None:
                real_result = self._real_result if self._real_result else cast('Result[Any]', self)
                if real_result._yield_per:
                    num_required = num = real_result._yield_per
                else:
                    rows = _manyrows(num)
                    num = len(rows)
                    assert make_row is not None
                    collect.extend(filterrows(make_row, rows, strategy, uniques))
                    num_required = num - len(collect)
            else:
                num_required = num
            assert num is not None
            while num_required:
                rows = _manyrows(num_required)
                if not rows:
                    break
                collect.extend(filterrows(make_row, rows, strategy, uniques))
                num_required = num - len(collect)
            if post_creational_filter:
                collect = [post_creational_filter(row) for row in collect]
            return collect
    else:

        def manyrows(self: ResultInternal[_R], num: Optional[int]) -> List[_R]:
            if num is None:
                real_result = self._real_result if self._real_result else cast('Result[Any]', self)
                num = real_result._yield_per
            rows: List[_InterimRowType[Any]] = self._fetchmany_impl(num)
            if make_row:
                rows = [make_row(row) for row in rows]
            if post_creational_filter:
                rows = [post_creational_filter(row) for row in rows]
            return rows
    return manyrows