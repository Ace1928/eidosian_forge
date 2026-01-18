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
@operation(Operation.FROM)
def unionByName(self, other: DataFrame, allowMissingColumns: bool=False):
    l_columns = self.columns
    r_columns = other.columns
    if not allowMissingColumns:
        l_expressions = l_columns
        r_expressions = l_columns
    else:
        l_expressions = []
        r_expressions = []
        r_columns_unused = copy(r_columns)
        for l_column in l_columns:
            l_expressions.append(l_column)
            if l_column in r_columns:
                r_expressions.append(l_column)
                r_columns_unused.remove(l_column)
            else:
                r_expressions.append(exp.alias_(exp.Null(), l_column, copy=False))
        for r_column in r_columns_unused:
            l_expressions.append(exp.alias_(exp.Null(), r_column, copy=False))
            r_expressions.append(r_column)
    r_df = other.copy()._convert_leaf_to_cte().select(*self._ensure_list_of_columns(r_expressions))
    l_df = self.copy()
    if allowMissingColumns:
        l_df = l_df._convert_leaf_to_cte().select(*self._ensure_list_of_columns(l_expressions))
    return l_df._set_operation(exp.Union, r_df, False)