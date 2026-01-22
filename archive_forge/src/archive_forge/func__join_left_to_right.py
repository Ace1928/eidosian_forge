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
def _join_left_to_right(self, entities_collection, left, right, onclause, prop, outerjoin, full):
    """given raw "left", "right", "onclause" parameters consumed from
        a particular key within _join(), add a real ORMJoin object to
        our _from_obj list (or augment an existing one)

        """
    if left is None:
        assert prop is None
        left, replace_from_obj_index, use_entity_index = self._join_determine_implicit_left_side(entities_collection, left, right, onclause)
    else:
        replace_from_obj_index, use_entity_index = self._join_place_explicit_left_side(entities_collection, left)
    if left is right:
        raise sa_exc.InvalidRequestError("Can't construct a join from %s to %s, they are the same entity" % (left, right))
    r_info, right, onclause = self._join_check_and_adapt_right_side(left, right, onclause, prop)
    if not r_info.is_selectable:
        extra_criteria = self._get_extra_criteria(r_info)
    else:
        extra_criteria = ()
    if replace_from_obj_index is not None:
        left_clause = self.from_clauses[replace_from_obj_index]
        self.from_clauses = self.from_clauses[:replace_from_obj_index] + [_ORMJoin(left_clause, right, onclause, isouter=outerjoin, full=full, _extra_criteria=extra_criteria)] + self.from_clauses[replace_from_obj_index + 1:]
    else:
        if use_entity_index is not None:
            assert isinstance(entities_collection[use_entity_index], _MapperEntity)
            left_clause = entities_collection[use_entity_index].selectable
        else:
            left_clause = left
        self.from_clauses = self.from_clauses + [_ORMJoin(left_clause, r_info, onclause, isouter=outerjoin, full=full, _extra_criteria=extra_criteria)]