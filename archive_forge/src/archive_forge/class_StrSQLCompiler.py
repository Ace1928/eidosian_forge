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
class StrSQLCompiler(SQLCompiler):
    """A :class:`.SQLCompiler` subclass which allows a small selection
    of non-standard SQL features to render into a string value.

    The :class:`.StrSQLCompiler` is invoked whenever a Core expression
    element is directly stringified without calling upon the
    :meth:`_expression.ClauseElement.compile` method.
    It can render a limited set
    of non-standard SQL constructs to assist in basic stringification,
    however for more substantial custom or dialect-specific SQL constructs,
    it will be necessary to make use of
    :meth:`_expression.ClauseElement.compile`
    directly.

    .. seealso::

        :ref:`faq_sql_expression_string`

    """

    def _fallback_column_name(self, column):
        return '<name unknown>'

    @util.preload_module('sqlalchemy.engine.url')
    def visit_unsupported_compilation(self, element, err, **kw):
        if element.stringify_dialect != 'default':
            url = util.preloaded.engine_url
            dialect = url.URL.create(element.stringify_dialect).get_dialect()()
            compiler = dialect.statement_compiler(dialect, None, _supporting_against=self)
            if not isinstance(compiler, StrSQLCompiler):
                return compiler.process(element, **kw)
        return super().visit_unsupported_compilation(element, err)

    def visit_getitem_binary(self, binary, operator, **kw):
        return '%s[%s]' % (self.process(binary.left, **kw), self.process(binary.right, **kw))

    def visit_json_getitem_op_binary(self, binary, operator, **kw):
        return self.visit_getitem_binary(binary, operator, **kw)

    def visit_json_path_getitem_op_binary(self, binary, operator, **kw):
        return self.visit_getitem_binary(binary, operator, **kw)

    def visit_sequence(self, seq, **kw):
        return '<next sequence value: %s>' % self.preparer.format_sequence(seq)

    def returning_clause(self, stmt: UpdateBase, returning_cols: Sequence[ColumnElement[Any]], *, populate_result_map: bool, **kw: Any) -> str:
        columns = [self._label_select_column(None, c, True, False, {}) for c in base._select_iterables(returning_cols)]
        return 'RETURNING ' + ', '.join(columns)

    def update_from_clause(self, update_stmt, from_table, extra_froms, from_hints, **kw):
        kw['asfrom'] = True
        return 'FROM ' + ', '.join((t._compiler_dispatch(self, fromhints=from_hints, **kw) for t in extra_froms))

    def delete_extra_from_clause(self, update_stmt, from_table, extra_froms, from_hints, **kw):
        kw['asfrom'] = True
        return ', ' + ', '.join((t._compiler_dispatch(self, fromhints=from_hints, **kw) for t in extra_froms))

    def visit_empty_set_expr(self, type_, **kw):
        return 'SELECT 1 WHERE 1!=1'

    def get_from_hint_text(self, table, text):
        return '[%s]' % text

    def visit_regexp_match_op_binary(self, binary, operator, **kw):
        return self._generate_generic_binary(binary, ' <regexp> ', **kw)

    def visit_not_regexp_match_op_binary(self, binary, operator, **kw):
        return self._generate_generic_binary(binary, ' <not regexp> ', **kw)

    def visit_regexp_replace_op_binary(self, binary, operator, **kw):
        return '<regexp replace>(%s, %s)' % (binary.left._compiler_dispatch(self, **kw), binary.right._compiler_dispatch(self, **kw))

    def visit_try_cast(self, cast, **kwargs):
        return 'TRY_CAST(%s AS %s)' % (cast.clause._compiler_dispatch(self, **kwargs), cast.typeclause._compiler_dispatch(self, **kwargs))