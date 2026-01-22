from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import CheckConstraint
from sqlalchemy import Column
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import Index
from sqlalchemy import MetaData
from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy import schema as sql_schema
from sqlalchemy import Table
from sqlalchemy import types as sqltypes
from sqlalchemy.sql.schema import SchemaEventTarget
from sqlalchemy.util import OrderedDict
from sqlalchemy.util import topological
from ..util import exc
from ..util.sqla_compat import _columns_for_constraint
from ..util.sqla_compat import _copy
from ..util.sqla_compat import _copy_expression
from ..util.sqla_compat import _ensure_scope_for_ddl
from ..util.sqla_compat import _fk_is_self_referential
from ..util.sqla_compat import _idx_table_bound_expressions
from ..util.sqla_compat import _insert_inline
from ..util.sqla_compat import _is_type_bound
from ..util.sqla_compat import _remove_column_from_collection
from ..util.sqla_compat import _resolve_for_variant
from ..util.sqla_compat import _select
from ..util.sqla_compat import constraint_name_defined
from ..util.sqla_compat import constraint_name_string
def _setup_dependencies_for_add_column(self, colname: str, insert_before: Optional[str], insert_after: Optional[str]) -> None:
    index_cols = self.existing_ordering
    col_indexes = {name: i for i, name in enumerate(index_cols)}
    if not self.partial_reordering:
        if insert_after:
            if not insert_before:
                if insert_after in col_indexes:
                    idx = col_indexes[insert_after] + 1
                    if idx < len(index_cols):
                        insert_before = index_cols[idx]
                else:
                    insert_before = dict(self.add_col_ordering)[insert_after]
        if insert_before:
            if not insert_after:
                if insert_before in col_indexes:
                    idx = col_indexes[insert_before] - 1
                    if idx >= 0:
                        insert_after = index_cols[idx]
                else:
                    insert_after = {b: a for a, b in self.add_col_ordering}[insert_before]
    if insert_before:
        self.add_col_ordering += ((colname, insert_before),)
    if insert_after:
        self.add_col_ordering += ((insert_after, colname),)
    if not self.partial_reordering and (not insert_before) and (not insert_after) and col_indexes:
        self.add_col_ordering += ((index_cols[-1], colname),)