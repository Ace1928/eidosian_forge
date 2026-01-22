from __future__ import annotations
import functools
import operator
import random
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import weakref
from . import characteristics
from . import cursor as _cursor
from . import interfaces
from .base import Connection
from .interfaces import CacheStats
from .interfaces import DBAPICursor
from .interfaces import Dialect
from .interfaces import ExecuteStyle
from .interfaces import ExecutionContext
from .reflection import ObjectKind
from .reflection import ObjectScope
from .. import event
from .. import exc
from .. import pool
from .. import util
from ..sql import compiler
from ..sql import dml
from ..sql import expression
from ..sql import type_api
from ..sql._typing import is_tuple_type
from ..sql.base import _NoArg
from ..sql.compiler import DDLCompiler
from ..sql.compiler import InsertmanyvaluesSentinelOpts
from ..sql.compiler import SQLCompiler
from ..sql.elements import quoted_name
from ..util.typing import Final
from ..util.typing import Literal
def _setup_dml_or_text_result(self):
    compiled = cast(SQLCompiler, self.compiled)
    strategy: ResultFetchStrategy = self.cursor_fetch_strategy
    if self.isinsert:
        if self.execute_style is ExecuteStyle.INSERTMANYVALUES and compiled.effective_returning:
            strategy = _cursor.FullyBufferedCursorFetchStrategy(self.cursor, initial_buffer=self._insertmanyvalues_rows, alternate_description=strategy.alternate_cursor_description)
        if compiled.postfetch_lastrowid:
            self.inserted_primary_key_rows = self._setup_ins_pk_from_lastrowid()
    if self._is_server_side and strategy is _cursor._DEFAULT_FETCH:
        strategy = _cursor.BufferedRowCursorFetchStrategy(self.cursor, self.execution_options)
    if strategy is _cursor._NO_CURSOR_DML:
        cursor_description = None
    else:
        cursor_description = strategy.alternate_cursor_description or self.cursor.description
    if cursor_description is None:
        strategy = _cursor._NO_CURSOR_DML
    elif self._num_sentinel_cols:
        assert self.execute_style is ExecuteStyle.INSERTMANYVALUES
        cursor_description = cursor_description[0:-self._num_sentinel_cols]
    result: _cursor.CursorResult[Any] = _cursor.CursorResult(self, strategy, cursor_description)
    if self.isinsert:
        if self._is_implicit_returning:
            rows = result.all()
            self.returned_default_rows = rows
            self.inserted_primary_key_rows = self._setup_ins_pk_from_implicit_returning(result, rows)
            assert result._metadata.returns_rows
            if self._is_supplemental_returning:
                result._rewind(rows)
            else:
                result._soft_close()
        elif not self._is_explicit_returning:
            result._soft_close()
    elif self._is_implicit_returning:
        rows = result.all()
        if rows:
            self.returned_default_rows = rows
        self._rowcount = len(rows)
        if self._is_supplemental_returning:
            result._rewind(rows)
        else:
            result._soft_close()
        assert result._metadata.returns_rows
    elif not result._metadata.returns_rows:
        if self._rowcount is None:
            self._rowcount = self.cursor.rowcount
        result._soft_close()
    elif self.isupdate or self.isdelete:
        if self._rowcount is None:
            self._rowcount = self.cursor.rowcount
    return result