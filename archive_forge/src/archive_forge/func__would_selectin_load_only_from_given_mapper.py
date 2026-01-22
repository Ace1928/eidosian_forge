from __future__ import annotations
from collections import deque
from functools import reduce
from itertools import chain
import sys
import threading
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Deque
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import exc as orm_exc
from . import instrumentation
from . import loading
from . import properties
from . import util as orm_util
from ._typing import _O
from .base import _class_to_mapper
from .base import _parse_mapper_argument
from .base import _state_mapper
from .base import PassiveFlag
from .base import state_str
from .interfaces import _MappedAttribute
from .interfaces import EXT_SKIP
from .interfaces import InspectionAttr
from .interfaces import MapperProperty
from .interfaces import ORMEntityColumnsClauseRole
from .interfaces import ORMFromClauseRole
from .interfaces import StrategizedProperty
from .path_registry import PathRegistry
from .. import event
from .. import exc as sa_exc
from .. import inspection
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..event import dispatcher
from ..event import EventTarget
from ..sql import base as sql_base
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import TableClause
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.cache_key import MemoizedHasCacheKey
from ..sql.elements import KeyedColumnElement
from ..sql.schema import Column
from ..sql.schema import Table
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import HasMemoized
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
def _would_selectin_load_only_from_given_mapper(self, super_mapper):
    """return True if this mapper would "selectin" polymorphic load based
        on the given super mapper, and not from a setting from a subclass.

        given::

            class A:
                ...

            class B(A):
                __mapper_args__ = {"polymorphic_load": "selectin"}

            class C(B):
                ...

            class D(B):
                __mapper_args__ = {"polymorphic_load": "selectin"}

        ``inspect(C)._would_selectin_load_only_from_given_mapper(inspect(B))``
        returns True, because C does selectin loading because of B's setting.

        OTOH, ``inspect(D)
        ._would_selectin_load_only_from_given_mapper(inspect(B))``
        returns False, because D does selectin loading because of its own
        setting; when we are doing a selectin poly load from B, we want to
        filter out D because it would already have its own selectin poly load
        set up separately.

        Added as part of #9373.

        """
    cache = self._would_selectinload_combinations_cache
    try:
        return cache[super_mapper]
    except KeyError:
        pass
    assert self.isa(super_mapper)
    mapper = super_mapper
    for m in self._iterate_to_target_viawpoly(mapper):
        if m.polymorphic_load == 'selectin':
            retval = m is super_mapper
            break
    else:
        retval = False
    cache[super_mapper] = retval
    return retval