from __future__ import annotations
import contextlib
from dataclasses import dataclass
from enum import auto
from enum import Flag
from enum import unique
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import Connection
from .base import Engine
from .. import exc
from .. import inspection
from .. import sql
from .. import util
from ..sql import operators
from ..sql import schema as sa_schema
from ..sql.cache_key import _ad_hoc_cache_key_from_args
from ..sql.elements import TextClause
from ..sql.type_api import TypeEngine
from ..sql.visitors import InternalTraversal
from ..util import topological
from ..util.typing import final
def _reflect_column(self, table: sa_schema.Table, col_d: ReflectedColumn, include_columns: Optional[Collection[str]], exclude_columns: Collection[str], cols_by_orig_name: Dict[str, sa_schema.Column[Any]]) -> None:
    orig_name = col_d['name']
    table.metadata.dispatch.column_reflect(self, table, col_d)
    table.dispatch.column_reflect(self, table, col_d)
    name = col_d['name']
    if include_columns and name not in include_columns or (exclude_columns and name in exclude_columns):
        return
    coltype = col_d['type']
    col_kw = {k: col_d[k] for k in ['nullable', 'autoincrement', 'quote', 'info', 'key', 'comment'] if k in col_d}
    if 'dialect_options' in col_d:
        col_kw.update(col_d['dialect_options'])
    colargs = []
    default: Any
    if col_d.get('default') is not None:
        default_text = col_d['default']
        assert default_text is not None
        if isinstance(default_text, TextClause):
            default = sa_schema.DefaultClause(default_text, _reflected=True)
        elif not isinstance(default_text, sa_schema.FetchedValue):
            default = sa_schema.DefaultClause(sql.text(default_text), _reflected=True)
        else:
            default = default_text
        colargs.append(default)
    if 'computed' in col_d:
        computed = sa_schema.Computed(**col_d['computed'])
        colargs.append(computed)
    if 'identity' in col_d:
        identity = sa_schema.Identity(**col_d['identity'])
        colargs.append(identity)
    cols_by_orig_name[orig_name] = col = sa_schema.Column(name, coltype, *colargs, **col_kw)
    if col.key in table.primary_key:
        col.primary_key = True
    table.append_column(col, replace_existing=True)