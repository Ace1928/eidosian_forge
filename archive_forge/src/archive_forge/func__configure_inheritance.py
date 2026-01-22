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
def _configure_inheritance(self):
    """Configure settings related to inheriting and/or inherited mappers
        being present."""
    self._inheriting_mappers = util.WeakSequence()
    if self.inherits:
        if not issubclass(self.class_, self.inherits.class_):
            raise sa_exc.ArgumentError("Class '%s' does not inherit from '%s'" % (self.class_.__name__, self.inherits.class_.__name__))
        self.dispatch._update(self.inherits.dispatch)
        if self.non_primary != self.inherits.non_primary:
            np = not self.non_primary and 'primary' or 'non-primary'
            raise sa_exc.ArgumentError("Inheritance of %s mapper for class '%s' is only allowed from a %s mapper" % (np, self.class_.__name__, np))
        if self.single:
            self.persist_selectable = self.inherits.persist_selectable
        elif self.local_table is not self.inherits.local_table:
            if self.concrete:
                self.persist_selectable = self.local_table
                for mapper in self.iterate_to_root():
                    if mapper.polymorphic_on is not None:
                        mapper._requires_row_aliasing = True
            else:
                if self.inherit_condition is None:
                    try:
                        self.inherit_condition = sql_util.join_condition(self.inherits.local_table, self.local_table)
                    except sa_exc.NoForeignKeysError as nfe:
                        assert self.inherits.local_table is not None
                        assert self.local_table is not None
                        raise sa_exc.NoForeignKeysError("Can't determine the inherit condition between inherited table '%s' and inheriting table '%s'; tables have no foreign key relationships established.  Please ensure the inheriting table has a foreign key relationship to the inherited table, or provide an 'on clause' using the 'inherit_condition' mapper argument." % (self.inherits.local_table.description, self.local_table.description)) from nfe
                    except sa_exc.AmbiguousForeignKeysError as afe:
                        assert self.inherits.local_table is not None
                        assert self.local_table is not None
                        raise sa_exc.AmbiguousForeignKeysError("Can't determine the inherit condition between inherited table '%s' and inheriting table '%s'; tables have more than one foreign key relationship established.  Please specify the 'on clause' using the 'inherit_condition' mapper argument." % (self.inherits.local_table.description, self.local_table.description)) from afe
                assert self.inherits.persist_selectable is not None
                self.persist_selectable = sql.join(self.inherits.persist_selectable, self.local_table, self.inherit_condition)
                fks = util.to_set(self.inherit_foreign_keys)
                self._inherits_equated_pairs = sql_util.criterion_as_pairs(self.persist_selectable.onclause, consider_as_foreign_keys=fks)
        else:
            self.persist_selectable = self.local_table
        if self.polymorphic_identity is None:
            self._identity_class = self.class_
            if not self.polymorphic_abstract and self.inherits.base_mapper.polymorphic_on is not None:
                util.warn(f"{self} does not indicate a 'polymorphic_identity', yet is part of an inheritance hierarchy that has a 'polymorphic_on' column of '{self.inherits.base_mapper.polymorphic_on}'. If this is an intermediary class that should not be instantiated, the class may either be left unmapped, or may include the 'polymorphic_abstract=True' parameter in its Mapper arguments. To leave the class unmapped when using Declarative, set the '__abstract__ = True' attribute on the class.")
        elif self.concrete:
            self._identity_class = self.class_
        else:
            self._identity_class = self.inherits._identity_class
        if self.version_id_col is None:
            self.version_id_col = self.inherits.version_id_col
            self.version_id_generator = self.inherits.version_id_generator
        elif self.inherits.version_id_col is not None and self.version_id_col is not self.inherits.version_id_col:
            util.warn("Inheriting version_id_col '%s' does not match inherited version_id_col '%s' and will not automatically populate the inherited versioning column. version_id_col should only be specified on the base-most mapper that includes versioning." % (self.version_id_col.description, self.inherits.version_id_col.description))
        self.polymorphic_map = self.inherits.polymorphic_map
        self.batch = self.inherits.batch
        self.inherits._inheriting_mappers.append(self)
        self.base_mapper = self.inherits.base_mapper
        self.passive_updates = self.inherits.passive_updates
        self.passive_deletes = self.inherits.passive_deletes or self.passive_deletes
        self._all_tables = self.inherits._all_tables
        if self.polymorphic_identity is not None:
            if self.polymorphic_identity in self.polymorphic_map:
                util.warn('Reassigning polymorphic association for identity %r from %r to %r: Check for duplicate use of %r as value for polymorphic_identity.' % (self.polymorphic_identity, self.polymorphic_map[self.polymorphic_identity], self, self.polymorphic_identity))
            self.polymorphic_map[self.polymorphic_identity] = self
        if self.polymorphic_load and self.concrete:
            raise sa_exc.ArgumentError('polymorphic_load is not currently supported with concrete table inheritance')
        if self.polymorphic_load == 'inline':
            self.inherits._add_with_polymorphic_subclass(self)
        elif self.polymorphic_load == 'selectin':
            pass
        elif self.polymorphic_load is not None:
            raise sa_exc.ArgumentError('unknown argument for polymorphic_load: %r' % self.polymorphic_load)
    else:
        self._all_tables = set()
        self.base_mapper = self
        assert self.local_table is not None
        self.persist_selectable = self.local_table
        if self.polymorphic_identity is not None:
            self.polymorphic_map[self.polymorphic_identity] = self
        self._identity_class = self.class_
    if self.persist_selectable is None:
        raise sa_exc.ArgumentError("Mapper '%s' does not have a persist_selectable specified." % self)