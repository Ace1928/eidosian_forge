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
@operation(Operation.SELECT)
def withColumn(self, colName: str, col: Column) -> DataFrame:
    col = self._ensure_and_normalize_col(col)
    existing_col_names = self.expression.named_selects
    existing_col_index = existing_col_names.index(colName) if colName in existing_col_names else None
    if existing_col_index:
        expression = self.expression.copy()
        expression.expressions[existing_col_index] = col.expression
        return self.copy(expression=expression)
    return self.copy().select(col.alias(colName), append=True)