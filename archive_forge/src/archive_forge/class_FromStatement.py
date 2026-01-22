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
class FromStatement(GroupedElement, Generative, TypedReturnsRows[_TP]):
    """Core construct that represents a load of ORM objects from various
    :class:`.ReturnsRows` and other classes including:

    :class:`.Select`, :class:`.TextClause`, :class:`.TextualSelect`,
    :class:`.CompoundSelect`, :class`.Insert`, :class:`.Update`,
    and in theory, :class:`.Delete`.

    """
    __visit_name__ = 'orm_from_statement'
    _compile_options = ORMFromStatementCompileState.default_compile_options
    _compile_state_factory = ORMFromStatementCompileState.create_for_statement
    _for_update_arg = None
    element: Union[ExecutableReturnsRows, TextClause]
    _adapt_on_names: bool
    _traverse_internals = [('_raw_columns', InternalTraversal.dp_clauseelement_list), ('element', InternalTraversal.dp_clauseelement)] + Executable._executable_traverse_internals
    _cache_key_traversal = _traverse_internals + [('_compile_options', InternalTraversal.dp_has_cache_key)]

    def __init__(self, entities: Iterable[_ColumnsClauseArgument[Any]], element: Union[ExecutableReturnsRows, TextClause], _adapt_on_names: bool=True):
        self._raw_columns = [coercions.expect(roles.ColumnsClauseRole, ent, apply_propagate_attrs=self, post_inspect=True) for ent in util.to_list(entities)]
        self.element = element
        self.is_dml = element.is_dml
        self._label_style = element._label_style if is_select_base(element) else None
        self._adapt_on_names = _adapt_on_names

    def _compiler_dispatch(self, compiler, **kw):
        """provide a fixed _compiler_dispatch method.

        This is roughly similar to using the sqlalchemy.ext.compiler
        ``@compiles`` extension.

        """
        compile_state = self._compile_state_factory(self, compiler, **kw)
        toplevel = not compiler.stack
        if toplevel:
            compiler.compile_state = compile_state
        return compiler.process(compile_state.statement, **kw)

    @property
    def column_descriptions(self):
        """Return a :term:`plugin-enabled` 'column descriptions' structure
        referring to the columns which are SELECTed by this statement.

        See the section :ref:`queryguide_inspection` for an overview
        of this feature.

        .. seealso::

            :ref:`queryguide_inspection` - ORM background

        """
        meth = cast(ORMSelectCompileState, SelectState.get_plugin_class(self)).get_column_descriptions
        return meth(self)

    def _ensure_disambiguated_names(self):
        return self

    def get_children(self, **kw):
        yield from itertools.chain.from_iterable((element._from_objects for element in self._raw_columns))
        yield from super().get_children(**kw)

    @property
    def _all_selected_columns(self):
        return self.element._all_selected_columns

    @property
    def _return_defaults(self):
        return self.element._return_defaults if is_dml(self.element) else None

    @property
    def _returning(self):
        return self.element._returning if is_dml(self.element) else None

    @property
    def _inline(self):
        return self.element._inline if is_insert_update(self.element) else None