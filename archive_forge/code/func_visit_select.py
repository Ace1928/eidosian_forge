from __future__ import annotations
import collections
import collections.abc as collections_abc
import contextlib
from enum import IntEnum
import functools
import itertools
import operator
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from . import base
from . import coercions
from . import crud
from . import elements
from . import functions
from . import operators
from . import roles
from . import schema
from . import selectable
from . import sqltypes
from . import util as sql_util
from ._typing import is_column_element
from ._typing import is_dml
from .base import _de_clone
from .base import _from_objects
from .base import _NONE_NAME
from .base import _SentinelDefaultCharacterization
from .base import Executable
from .base import NO_ARG
from .elements import ClauseElement
from .elements import quoted_name
from .schema import Column
from .sqltypes import TupleType
from .type_api import TypeEngine
from .visitors import prefix_anon_map
from .visitors import Visitable
from .. import exc
from .. import util
from ..util import FastIntFlag
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import TypedDict
def visit_select(self, select_stmt, asfrom=False, insert_into=False, fromhints=None, compound_index=None, select_wraps_for=None, lateral=False, from_linter=None, **kwargs):
    assert select_wraps_for is None, 'SQLAlchemy 1.4 requires use of the translate_select_structure hook for structural translations of SELECT objects'
    kwargs['within_columns_clause'] = False
    compile_state = select_stmt._compile_state_factory(select_stmt, self, **kwargs)
    kwargs['ambiguous_table_name_map'] = compile_state._ambiguous_table_name_map
    select_stmt = compile_state.statement
    toplevel = not self.stack
    if toplevel and (not self.compile_state):
        self.compile_state = compile_state
    is_embedded_select = compound_index is not None or insert_into
    if self.translate_select_structure:
        new_select_stmt = self.translate_select_structure(select_stmt, asfrom=asfrom, **kwargs)
        if new_select_stmt is not select_stmt:
            compile_state_wraps_for = compile_state
            select_wraps_for = select_stmt
            select_stmt = new_select_stmt
            compile_state = select_stmt._compile_state_factory(select_stmt, self, **kwargs)
            select_stmt = compile_state.statement
    entry = self._default_stack_entry if toplevel else self.stack[-1]
    populate_result_map = need_column_expressions = toplevel or entry.get('need_result_map_for_compound', False) or entry.get('need_result_map_for_nested', False)
    if compound_index:
        populate_result_map = False
    if not populate_result_map and 'add_to_result_map' in kwargs:
        del kwargs['add_to_result_map']
    froms = self._setup_select_stack(select_stmt, compile_state, entry, asfrom, lateral, compound_index)
    column_clause_args = kwargs.copy()
    column_clause_args.update({'within_label_clause': False, 'within_columns_clause': False})
    text = 'SELECT '
    if select_stmt._hints:
        hint_text, byfrom = self._setup_select_hints(select_stmt)
        if hint_text:
            text += hint_text + ' '
    else:
        byfrom = None
    if select_stmt._independent_ctes:
        self._dispatch_independent_ctes(select_stmt, kwargs)
    if select_stmt._prefixes:
        text += self._generate_prefixes(select_stmt, select_stmt._prefixes, **kwargs)
    text += self.get_select_precolumns(select_stmt, **kwargs)
    inner_columns = [c for c in [self._label_select_column(select_stmt, column, populate_result_map, asfrom, column_clause_args, name=name, proxy_name=proxy_name, fallback_label_name=fallback_label_name, column_is_repeated=repeated, need_column_expressions=need_column_expressions) for name, proxy_name, fallback_label_name, column, repeated in compile_state.columns_plus_names] if c is not None]
    if populate_result_map and select_wraps_for is not None:
        translate = dict(zip([name for key, proxy_name, fallback_label_name, name, repeated in compile_state.columns_plus_names], [name for key, proxy_name, fallback_label_name, name, repeated in compile_state_wraps_for.columns_plus_names]))
        self._result_columns = [ResultColumnsEntry(key, name, tuple((translate.get(o, o) for o in obj)), type_) for key, name, obj, type_ in self._result_columns]
    text = self._compose_select_body(text, select_stmt, compile_state, inner_columns, froms, byfrom, toplevel, kwargs)
    if select_stmt._statement_hints:
        per_dialect = [ht for dialect_name, ht in select_stmt._statement_hints if dialect_name in ('*', self.dialect.name)]
        if per_dialect:
            text += ' ' + self.get_statement_hint_text(per_dialect)
    if self.ctes and (not is_embedded_select or toplevel):
        nesting_level = len(self.stack) if not toplevel else None
        text = self._render_cte_clause(nesting_level=nesting_level) + text
    if select_stmt._suffixes:
        text += ' ' + self._generate_prefixes(select_stmt, select_stmt._suffixes, **kwargs)
    self.stack.pop(-1)
    return text