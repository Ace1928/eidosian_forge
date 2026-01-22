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
def _adjust_self_columns_for_partial_reordering(self) -> None:
    pairs = set()
    col_by_idx = list(self.columns)
    if self.partial_reordering:
        for tuple_ in self.partial_reordering:
            for index, elem in enumerate(tuple_):
                if index > 0:
                    pairs.add((tuple_[index - 1], elem))
    else:
        for index, elem in enumerate(self.existing_ordering):
            if index > 0:
                pairs.add((col_by_idx[index - 1], elem))
    pairs.update(self.add_col_ordering)
    pairs_list = [p for p in pairs if p[0] != p[1]]
    sorted_ = list(topological.sort(pairs_list, col_by_idx, deterministic_order=True))
    self.columns = OrderedDict(((k, self.columns[k]) for k in sorted_))
    self.column_transfers = OrderedDict(((k, self.column_transfers[k]) for k in sorted_))