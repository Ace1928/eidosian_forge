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
def withColumnRenamed(self, existing: str, new: str):
    expression = self.expression.copy()
    existing_columns = [expression for expression in expression.expressions if expression.alias_or_name == existing]
    if not existing_columns:
        raise ValueError("Tried to rename a column that doesn't exist")
    for existing_column in existing_columns:
        if isinstance(existing_column, exp.Column):
            existing_column.replace(exp.alias_(existing_column, new))
        else:
            existing_column.set('alias', exp.to_identifier(new))
    return self.copy(expression=expression)