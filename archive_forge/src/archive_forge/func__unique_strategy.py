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
@HasMemoized.memoized_attribute
def _unique_strategy(self) -> _UniqueFilterStateType:
    assert self._unique_filter_state is not None
    uniques, strategy = self._unique_filter_state
    real_result = self._real_result if self._real_result is not None else cast('Result[Any]', self)
    if not strategy and self._metadata._unique_filters:
        if real_result._source_supports_scalars and (not self._generate_rows):
            strategy = self._metadata._unique_filters[0]
        else:
            filters = self._metadata._unique_filters
            if self._metadata._tuplefilter:
                filters = self._metadata._tuplefilter(filters)
            strategy = operator.methodcaller('_filter_on_values', filters)
    return (uniques, strategy)