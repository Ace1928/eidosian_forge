from __future__ import annotations
import enum
from itertools import zip_longest
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from .visitors import anon_map
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import util
from ..inspection import inspect
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
@classmethod
def _generate_cache_key_for_object(cls, obj: HasCacheKey) -> Optional[CacheKey]:
    bindparams: List[BindParameter[Any]] = []
    _anon_map = anon_map()
    key = obj._gen_cache_key(_anon_map, bindparams)
    if NO_CACHE in _anon_map:
        return None
    else:
        assert key is not None
        return CacheKey(key, bindparams)