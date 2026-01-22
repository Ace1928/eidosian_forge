from __future__ import annotations
import collections
from collections import abc
import dataclasses
import inspect as _py_inspect
import itertools
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import strategy_options
from ._typing import insp_is_aliased_class
from ._typing import is_has_collection_adapter
from .base import _DeclarativeMapped
from .base import _is_mapped_class
from .base import class_mapper
from .base import DynamicMapped
from .base import LoaderCallableStatus
from .base import PassiveFlag
from .base import state_str
from .base import WriteOnlyMapped
from .interfaces import _AttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .interfaces import PropComparator
from .interfaces import RelationshipDirection
from .interfaces import StrategizedProperty
from .util import _orm_annotate
from .util import _orm_deannotate
from .util import CascadeOptions
from .. import exc as sa_exc
from .. import Exists
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..inspection import inspect
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import visitors
from ..sql._typing import _ColumnExpressionArgument
from ..sql._typing import _HasClauseElement
from ..sql.annotation import _safe_annotate
from ..sql.elements import ColumnClause
from ..sql.elements import ColumnElement
from ..sql.util import _deep_annotate
from ..sql.util import _deep_deannotate
from ..sql.util import _shallow_annotate
from ..sql.util import adapt_criterion_to_null
from ..sql.util import ClauseAdapter
from ..sql.util import join_condition
from ..sql.util import selectables_overlap
from ..sql.util import visit_binary_product
from ..util.typing import de_optionalize_union_types
from ..util.typing import Literal
from ..util.typing import resolve_name_to_real_class_name
@util.preload_module('sqlalchemy.orm.mapper')
def _setup_entity(self, __argument: Any=None) -> None:
    if 'entity' in self.__dict__:
        return
    mapperlib = util.preloaded.orm_mapper
    if __argument:
        argument = __argument
    else:
        argument = self.argument
    resolved_argument: _ExternalEntityType[Any]
    if isinstance(argument, str):
        resolved_argument = cast('_ExternalEntityType[Any]', self._clsregistry_resolve_name(argument)())
    elif callable(argument) and (not isinstance(argument, (type, mapperlib.Mapper))):
        resolved_argument = argument()
    else:
        resolved_argument = argument
    entity: _InternalEntityType[Any]
    if isinstance(resolved_argument, type):
        entity = class_mapper(resolved_argument, configure=False)
    else:
        try:
            entity = inspect(resolved_argument)
        except sa_exc.NoInspectionAvailable:
            entity = None
        if not hasattr(entity, 'mapper'):
            raise sa_exc.ArgumentError("relationship '%s' expects a class or a mapper argument (received: %s)" % (self.key, type(resolved_argument)))
    self.entity = entity
    self.target = self.entity.persist_selectable