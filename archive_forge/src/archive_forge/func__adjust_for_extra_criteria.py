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
def _adjust_for_extra_criteria(self):
    """Apply extra criteria filtering.

        For all distinct single-table-inheritance mappers represented in
        the columns clause of this query, as well as the "select from entity",
        add criterion to the WHERE
        clause of the given QueryContext such that only the appropriate
        subtypes are selected from the total results.

        Additionally, add WHERE criteria originating from LoaderCriteriaOptions
        associated with the global context.

        """
    for fromclause in self.from_clauses:
        ext_info = fromclause._annotations.get('parententity', None)
        if ext_info and (ext_info.mapper._single_table_criterion is not None or ('additional_entity_criteria', ext_info.mapper) in self.global_attributes) and (ext_info not in self.extra_criteria_entities):
            self.extra_criteria_entities[ext_info] = (ext_info, ext_info._adapter if ext_info.is_aliased_class else None)
    search = set(self.extra_criteria_entities.values())
    for ext_info, adapter in search:
        if ext_info in self._join_entities:
            continue
        single_crit = ext_info.mapper._single_table_criterion
        if self.compile_options._for_refresh_state:
            additional_entity_criteria = []
        else:
            additional_entity_criteria = self._get_extra_criteria(ext_info)
        if single_crit is not None:
            additional_entity_criteria += (single_crit,)
        current_adapter = self._get_current_adapter()
        for crit in additional_entity_criteria:
            if adapter:
                crit = adapter.traverse(crit)
            if current_adapter:
                crit = sql_util._deep_annotate(crit, {'_orm_adapt': True})
                crit = current_adapter(crit, False)
            self._where_criteria += (crit,)