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
@classmethod
def _deserialize_path(cls, path: _SerializedPath) -> _PathRepresentation:

    def _deserialize_mapper_token(mcls: Any) -> Any:
        return orm_base._inspect_mapped_class(mcls, configure=True) if mcls not in PathToken._intern else PathToken._intern[mcls]

    def _deserialize_key_token(mcls: Any, key: Any) -> Any:
        if key is None:
            return None
        elif key in PathToken._intern:
            return PathToken._intern[key]
        else:
            mp = orm_base._inspect_mapped_class(mcls, configure=True)
            assert mp is not None
            return mp.attrs[key]
    p = tuple(chain(*[(_deserialize_mapper_token(mcls), _deserialize_key_token(mcls, key)) for mcls, key in path]))
    if p and p[-1] is None:
        p = p[0:-1]
    return p