from __future__ import annotations
import collections.abc as collections_abc
import inspect
import itertools
import operator
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import cache_key as _cache_key
from . import coercions
from . import elements
from . import roles
from . import schema
from . import visitors
from .base import _clone
from .base import Executable
from .base import Options
from .cache_key import CacheConst
from .operators import ColumnOperators
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
def _cache_key_getter_track_on(self, idx, elem):
    """Return a getter that will extend a cache key with new entries
        from the "track_on" parameter passed to a :class:`.LambdaElement`.

        """
    if isinstance(elem, tuple):

        def get(closure, opts, anon_map, bindparams):
            return tuple((tup_elem._gen_cache_key(anon_map, bindparams) for tup_elem in opts.track_on[idx]))
    elif isinstance(elem, _cache_key.HasCacheKey):

        def get(closure, opts, anon_map, bindparams):
            return opts.track_on[idx]._gen_cache_key(anon_map, bindparams)
    else:

        def get(closure, opts, anon_map, bindparams):
            return opts.track_on[idx]
    return get