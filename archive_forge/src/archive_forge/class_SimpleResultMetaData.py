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
class SimpleResultMetaData(ResultMetaData):
    """result metadata for in-memory collections."""
    __slots__ = ('_keys', '_keymap', '_processors', '_tuplefilter', '_translated_indexes', '_unique_filters', '_key_to_index')
    _keys: Sequence[str]

    def __init__(self, keys: Sequence[str], extra: Optional[Sequence[Any]]=None, _processors: Optional[_ProcessorsType]=None, _tuplefilter: Optional[_TupleGetterType]=None, _translated_indexes: Optional[Sequence[int]]=None, _unique_filters: Optional[Sequence[Callable[[Any], Any]]]=None):
        self._keys = list(keys)
        self._tuplefilter = _tuplefilter
        self._translated_indexes = _translated_indexes
        self._unique_filters = _unique_filters
        if extra:
            recs_names = [((name,) + (extras if extras else ()), (index, name, extras)) for index, (name, extras) in enumerate(zip(self._keys, extra))]
        else:
            recs_names = [((name,), (index, name, ())) for index, name in enumerate(self._keys)]
        self._keymap = {key: rec for keys, rec in recs_names for key in keys}
        self._processors = _processors
        self._key_to_index = self._make_key_to_index(self._keymap, 0)

    def _has_key(self, key: object) -> bool:
        return key in self._keymap

    def _for_freeze(self) -> ResultMetaData:
        unique_filters = self._unique_filters
        if unique_filters and self._tuplefilter:
            unique_filters = self._tuplefilter(unique_filters)
        return SimpleResultMetaData(self._keys, extra=[self._keymap[key][2] for key in self._keys], _unique_filters=unique_filters)

    def __getstate__(self) -> Dict[str, Any]:
        return {'_keys': self._keys, '_translated_indexes': self._translated_indexes}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if state['_translated_indexes']:
            _translated_indexes = state['_translated_indexes']
            _tuplefilter = tuplegetter(*_translated_indexes)
        else:
            _translated_indexes = _tuplefilter = None
        self.__init__(state['_keys'], _translated_indexes=_translated_indexes, _tuplefilter=_tuplefilter)

    def _index_for_key(self, key: Any, raiseerr: bool=True) -> int:
        if int in key.__class__.__mro__:
            key = self._keys[key]
        try:
            rec = self._keymap[key]
        except KeyError as ke:
            rec = self._key_fallback(key, ke, raiseerr)
        return rec[0]

    def _indexes_for_keys(self, keys: Sequence[Any]) -> Sequence[int]:
        return [self._keymap[key][0] for key in keys]

    def _metadata_for_keys(self, keys: Sequence[Any]) -> Iterator[_KeyMapRecType]:
        for key in keys:
            if int in key.__class__.__mro__:
                key = self._keys[key]
            try:
                rec = self._keymap[key]
            except KeyError as ke:
                rec = self._key_fallback(key, ke, True)
            yield rec

    def _reduce(self, keys: Sequence[Any]) -> ResultMetaData:
        try:
            metadata_for_keys = [self._keymap[self._keys[key] if int in key.__class__.__mro__ else key] for key in keys]
        except KeyError as ke:
            self._key_fallback(ke.args[0], ke, True)
        indexes: Sequence[int]
        new_keys: Sequence[str]
        extra: Sequence[Any]
        indexes, new_keys, extra = zip(*metadata_for_keys)
        if self._translated_indexes:
            indexes = [self._translated_indexes[idx] for idx in indexes]
        tup = tuplegetter(*indexes)
        new_metadata = SimpleResultMetaData(new_keys, extra=extra, _tuplefilter=tup, _translated_indexes=indexes, _processors=self._processors, _unique_filters=self._unique_filters)
        return new_metadata