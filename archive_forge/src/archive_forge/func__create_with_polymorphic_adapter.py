from __future__ import annotations
import itertools
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import interfaces
from . import loading
from .base import _is_aliased_class
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .path_registry import PathRegistry
from .util import _entity_corresponds_to
from .util import _ORMJoin
from .util import _TraceAdaptRole
from .util import AliasedClass
from .util import Bundle
from .util import ORMAdapter
from .util import ORMStatementAdapter
from .. import exc as sa_exc
from .. import future
from .. import inspect
from .. import sql
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _TP
from ..sql._typing import is_dml
from ..sql._typing import is_insert_update
from ..sql._typing import is_select_base
from ..sql.base import _select_iterables
from ..sql.base import CacheableOptions
from ..sql.base import CompileState
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.base import Options
from ..sql.dml import UpdateBase
from ..sql.elements import GroupedElement
from ..sql.elements import TextClause
from ..sql.selectable import CompoundSelectState
from ..sql.selectable import LABEL_STYLE_DISAMBIGUATE_ONLY
from ..sql.selectable import LABEL_STYLE_NONE
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
from ..sql.selectable import SelectLabelStyle
from ..sql.selectable import SelectState
from ..sql.selectable import TypedReturnsRows
from ..sql.visitors import InternalTraversal
def _create_with_polymorphic_adapter(self, ext_info, selectable):
    """given MapperEntity or ORMColumnEntity, setup polymorphic loading
        if called for by the Mapper.

        As of #8168 in 2.0.0rc1, polymorphic adapters, which greatly increase
        the complexity of the query creation process, are not used at all
        except in the quasi-legacy cases of with_polymorphic referring to an
        alias and/or subquery. This would apply to concrete polymorphic
        loading, and joined inheritance where a subquery is
        passed to with_polymorphic (which is completely unnecessary in modern
        use).

        """
    if not ext_info.is_aliased_class and ext_info.mapper.persist_selectable not in self._polymorphic_adapters:
        for mp in ext_info.mapper.iterate_to_root():
            self._mapper_loads_polymorphically_with(mp, ORMAdapter(_TraceAdaptRole.WITH_POLYMORPHIC_ADAPTER, mp, equivalents=mp._equivalent_columns, selectable=selectable))