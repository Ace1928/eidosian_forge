from __future__ import annotations
from functools import reduce
from itertools import chain
import logging
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import base as orm_base
from ._typing import insp_is_mapper_property
from .. import exc
from .. import util
from ..sql import visitors
from ..sql.cache_key import HasCacheKey
class CachingEntityRegistry(AbstractEntityRegistry):
    __slots__ = ('_cache',)
    inherit_cache = True

    def __init__(self, parent: Union[RootRegistry, PropRegistry], entity: _InternalEntityType[Any]):
        super().__init__(parent, entity)
        self._cache = _ERDict(self)

    def pop(self, key: Any, default: Any) -> Any:
        return self._cache.pop(key, default)

    def _getitem(self, entity: Any) -> Any:
        if isinstance(entity, (int, slice)):
            return self.path[entity]
        elif isinstance(entity, PathToken):
            return TokenRegistry(self, entity)
        else:
            return self._cache[entity]
    if not TYPE_CHECKING:
        __getitem__ = _getitem