from __future__ import annotations
import functools
import logging
import typing as t
import zlib
from copy import copy
import sqlglot
from sqlglot import Dialect, expressions as exp
from sqlglot.dataframe.sql import functions as F
from sqlglot.dataframe.sql.column import Column
from sqlglot.dataframe.sql.group import GroupedData
from sqlglot.dataframe.sql.normalize import normalize
from sqlglot.dataframe.sql.operations import Operation, operation
from sqlglot.dataframe.sql.readwriter import DataFrameWriter
from sqlglot.dataframe.sql.transforms import replace_id_value
from sqlglot.dataframe.sql.util import get_tables_from_expression_with_join
from sqlglot.dataframe.sql.window import Window
from sqlglot.helper import ensure_list, object_to_dict, seq_get
def _convert_leaf_to_cte(self, sequence_id: t.Optional[str]=None) -> DataFrame:
    df = self._resolve_pending_hints()
    sequence_id = sequence_id or df.sequence_id
    expression = df.expression.copy()
    cte_expression, cte_name = df._create_cte_from_expression(expression=expression, sequence_id=sequence_id)
    new_expression = df._add_ctes_to_expression(exp.Select(), expression.ctes + [cte_expression])
    sel_columns = df._get_outer_select_columns(cte_expression)
    new_expression = new_expression.from_(cte_name).select(*[x.alias_or_name for x in sel_columns])
    return df.copy(expression=new_expression, sequence_id=sequence_id)