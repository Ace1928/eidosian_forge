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
class ResultMetaData:
    """Base for metadata about result rows."""
    __slots__ = ()
    _tuplefilter: Optional[_TupleGetterType] = None
    _translated_indexes: Optional[Sequence[int]] = None
    _unique_filters: Optional[Sequence[Callable[[Any], Any]]] = None
    _keymap: _KeyMapType
    _keys: Sequence[str]
    _processors: Optional[_ProcessorsType]
    _key_to_index: Mapping[_KeyType, int]

    @property
    def keys(self) -> RMKeyView:
        return RMKeyView(self)

    def _has_key(self, key: object) -> bool:
        raise NotImplementedError()

    def _for_freeze(self) -> ResultMetaData:
        raise NotImplementedError()

    @overload
    def _key_fallback(self, key: Any, err: Optional[Exception], raiseerr: Literal[True]=...) -> NoReturn:
        ...

    @overload
    def _key_fallback(self, key: Any, err: Optional[Exception], raiseerr: Literal[False]=...) -> None:
        ...

    @overload
    def _key_fallback(self, key: Any, err: Optional[Exception], raiseerr: bool=...) -> Optional[NoReturn]:
        ...

    def _key_fallback(self, key: Any, err: Optional[Exception], raiseerr: bool=True) -> Optional[NoReturn]:
        assert raiseerr
        raise KeyError(key) from err

    def _raise_for_ambiguous_column_name(self, rec: _KeyMapRecType) -> NoReturn:
        raise NotImplementedError('ambiguous column name logic is implemented for CursorResultMetaData')

    def _index_for_key(self, key: _KeyIndexType, raiseerr: bool) -> Optional[int]:
        raise NotImplementedError()

    def _indexes_for_keys(self, keys: Sequence[_KeyIndexType]) -> Sequence[int]:
        raise NotImplementedError()

    def _metadata_for_keys(self, keys: Sequence[_KeyIndexType]) -> Iterator[_KeyMapRecType]:
        raise NotImplementedError()

    def _reduce(self, keys: Sequence[_KeyIndexType]) -> ResultMetaData:
        raise NotImplementedError()

    def _getter(self, key: Any, raiseerr: bool=True) -> Optional[Callable[[Row[Any]], Any]]:
        index = self._index_for_key(key, raiseerr)
        if index is not None:
            return operator.itemgetter(index)
        else:
            return None

    def _row_as_tuple_getter(self, keys: Sequence[_KeyIndexType]) -> _TupleGetterType:
        indexes = self._indexes_for_keys(keys)
        return tuplegetter(*indexes)

    def _make_key_to_index(self, keymap: Mapping[_KeyType, Sequence[Any]], index: int) -> Mapping[_KeyType, int]:
        return {key: rec[index] for key, rec in keymap.items() if rec[index] is not None}

    def _key_not_found(self, key: Any, attr_error: bool) -> NoReturn:
        if key in self._keymap:
            self._raise_for_ambiguous_column_name(self._keymap[key])
        elif attr_error:
            try:
                self._key_fallback(key, None)
            except KeyError as ke:
                raise AttributeError(ke.args[0]) from ke
        else:
            self._key_fallback(key, None)

    @property
    def _effective_processors(self) -> Optional[_ProcessorsType]:
        if not self._processors or NONE_SET.issuperset(self._processors):
            return None
        else:
            return self._processors