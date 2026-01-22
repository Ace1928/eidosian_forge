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
def _row_getter(self) -> Optional[Callable[..., _R]]:
    real_result: Result[Any] = self._real_result if self._real_result else cast('Result[Any]', self)
    if real_result._source_supports_scalars:
        if not self._generate_rows:
            return None
        else:
            _proc = Row

            def process_row(metadata: ResultMetaData, processors: Optional[_ProcessorsType], key_to_index: Mapping[_KeyType, int], scalar_obj: Any) -> Row[Any]:
                return _proc(metadata, processors, key_to_index, (scalar_obj,))
    else:
        process_row = Row
    metadata = self._metadata
    key_to_index = metadata._key_to_index
    processors = metadata._effective_processors
    tf = metadata._tuplefilter
    if tf and (not real_result._source_supports_scalars):
        if processors:
            processors = tf(processors)
        _make_row_orig: Callable[..., _R] = functools.partial(process_row, metadata, processors, key_to_index)
        fixed_tf = tf

        def make_row(row: _InterimRowType[Row[Any]]) -> _R:
            return _make_row_orig(fixed_tf(row))
    else:
        make_row = functools.partial(process_row, metadata, processors, key_to_index)
    if real_result._row_logging_fn:
        _log_row = real_result._row_logging_fn
        _make_row = make_row

        def make_row(row: _InterimRowType[Row[Any]]) -> _R:
            return _log_row(_make_row(row))
    return make_row