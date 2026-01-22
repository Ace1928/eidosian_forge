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
@classmethod
def _add_ctes_to_expression(cls, expression: exp.Select, ctes: t.List[exp.CTE]) -> exp.Select:
    expression = expression.copy()
    with_expression = expression.args.get('with')
    if with_expression:
        existing_ctes = with_expression.expressions
        existsing_cte_names = {x.alias_or_name for x in existing_ctes}
        for cte in ctes:
            if cte.alias_or_name not in existsing_cte_names:
                existing_ctes.append(cte)
    else:
        existing_ctes = ctes
    expression.set('with', exp.With(expressions=existing_ctes))
    return expression