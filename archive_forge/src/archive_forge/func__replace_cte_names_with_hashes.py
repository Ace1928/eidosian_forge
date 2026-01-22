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
def _replace_cte_names_with_hashes(self, expression: exp.Select):
    replacement_mapping = {}
    for cte in expression.ctes:
        old_name_id = cte.args['alias'].this
        new_hashed_id = exp.to_identifier(self._create_hash_from_expression(cte.this), quoted=old_name_id.args['quoted'])
        replacement_mapping[old_name_id] = new_hashed_id
        expression = expression.transform(replace_id_value, replacement_mapping).assert_is(exp.Select)
    return expression