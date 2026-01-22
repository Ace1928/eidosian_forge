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
def _add_reverse_property(self, key: str) -> None:
    other = self.mapper.get_property(key, _configure_mappers=False)
    if not isinstance(other, RelationshipProperty):
        raise sa_exc.InvalidRequestError("back_populates on relationship '%s' refers to attribute '%s' that is not a relationship.  The back_populates parameter should refer to the name of a relationship on the target class." % (self, other))
    self._check_sync_backref(self, other)
    self._check_sync_backref(other, self)
    self._reverse_property.add(other)
    other._reverse_property.add(self)
    other._setup_entity()
    if not other.mapper.common_parent(self.parent):
        raise sa_exc.ArgumentError('reverse_property %r on relationship %s references relationship %s, which does not reference mapper %s' % (key, self, other, self.parent))
    if other._configure_started and self.direction in (ONETOMANY, MANYTOONE) and (self.direction == other.direction):
        raise sa_exc.ArgumentError('%s and back-reference %s are both of the same direction %r.  Did you mean to set remote_side on the many-to-one side ?' % (other, self, self.direction))