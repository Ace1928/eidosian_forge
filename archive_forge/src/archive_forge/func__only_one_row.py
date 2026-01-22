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
def _only_one_row(self, raise_for_second_row: bool, raise_for_none: bool, scalar: bool) -> Optional[_R]:
    onerow = self._fetchone_impl
    row: Optional[_InterimRowType[Any]] = onerow(hard_close=True)
    if row is None:
        if raise_for_none:
            raise exc.NoResultFound('No row was found when one was required')
        else:
            return None
    if scalar and self._source_supports_scalars:
        self._generate_rows = False
        make_row = None
    else:
        make_row = self._row_getter
    try:
        row = make_row(row) if make_row else row
    except:
        self._soft_close(hard=True)
        raise
    if raise_for_second_row:
        if self._unique_filter_state:
            uniques, strategy = self._unique_strategy
            existing_row_hash = strategy(row) if strategy else row
            while True:
                next_row: Any = onerow(hard_close=True)
                if next_row is None:
                    next_row = _NO_ROW
                    break
                try:
                    next_row = make_row(next_row) if make_row else next_row
                    if strategy:
                        assert next_row is not _NO_ROW
                        if existing_row_hash == strategy(next_row):
                            continue
                    elif row == next_row:
                        continue
                    break
                except:
                    self._soft_close(hard=True)
                    raise
        else:
            next_row = onerow(hard_close=True)
            if next_row is None:
                next_row = _NO_ROW
        if next_row is not _NO_ROW:
            self._soft_close(hard=True)
            raise exc.MultipleResultsFound('Multiple rows were found when exactly one was required' if raise_for_none else 'Multiple rows were found when one or none was required')
    else:
        next_row = _NO_ROW
        self._soft_close(hard=True)
    if not scalar:
        post_creational_filter = self._post_creational_filter
        if post_creational_filter:
            row = post_creational_filter(row)
    if scalar and make_row:
        return row[0]
    else:
        return row