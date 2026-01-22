from __future__ import annotations
import collections.abc as collections_abc
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import interfaces
from . import loading
from . import util as orm_util
from ._typing import _O
from .base import _assertions
from .context import _column_descriptions
from .context import _determine_last_joined_entity
from .context import _legacy_filter_by_entity_zero
from .context import FromStatement
from .context import ORMCompileState
from .context import QueryContext
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .util import AliasedClass
from .util import object_mapper
from .util import with_parent
from .. import exc as sa_exc
from .. import inspect
from .. import inspection
from .. import log
from .. import sql
from .. import util
from ..engine import Result
from ..engine import Row
from ..event import dispatcher
from ..event import EventTarget
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import Select
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _FromClauseArgument
from ..sql._typing import _TP
from ..sql.annotation import SupportsCloneAnnotations
from ..sql.base import _entity_namespace_key
from ..sql.base import _generative
from ..sql.base import _NoArg
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.elements import BooleanClauseList
from ..sql.expression import Exists
from ..sql.selectable import _MemoizedSelectEntities
from ..sql.selectable import _SelectFromElements
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import HasHints
from ..sql.selectable import HasPrefixes
from ..sql.selectable import HasSuffixes
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import SelectLabelStyle
from ..util.typing import Literal
from ..util.typing import Self
def _statement_20(self, for_statement: bool=False, use_legacy_query_style: bool=True) -> Union[Select[_T], FromStatement[_T]]:
    if self.dispatch.before_compile:
        for fn in self.dispatch.before_compile:
            new_query = fn(self)
            if new_query is not None and new_query is not self:
                self = new_query
                if not fn._bake_ok:
                    self._compile_options += {'_bake_ok': False}
    compile_options = self._compile_options
    compile_options += {'_for_statement': for_statement, '_use_legacy_query_style': use_legacy_query_style}
    stmt: Union[Select[_T], FromStatement[_T]]
    if self._statement is not None:
        stmt = FromStatement(self._raw_columns, self._statement)
        stmt.__dict__.update(_with_options=self._with_options, _with_context_options=self._with_context_options, _compile_options=compile_options, _execution_options=self._execution_options, _propagate_attrs=self._propagate_attrs)
    else:
        stmt = Select._create_raw_select(**self.__dict__)
        stmt.__dict__.update(_label_style=self._label_style, _compile_options=compile_options, _propagate_attrs=self._propagate_attrs)
        stmt.__dict__.pop('session', None)
    if 'compile_state_plugin' not in stmt._propagate_attrs:
        stmt._propagate_attrs = stmt._propagate_attrs.union({'compile_state_plugin': 'orm', 'plugin_subject': None})
    return stmt