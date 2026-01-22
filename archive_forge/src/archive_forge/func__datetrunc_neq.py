from __future__ import annotations
import datetime
import functools
import itertools
import typing as t
from collections import deque
from decimal import Decimal
from functools import reduce
import sqlglot
from sqlglot import Dialect, exp
from sqlglot.helper import first, merge_ranges, while_changing
from sqlglot.optimizer.scope import find_all_in_scope, walk_in_scope
def _datetrunc_neq(left: exp.Expression, date: datetime.date, unit: str, dialect: Dialect, target_type: t.Optional[exp.DataType]) -> t.Optional[exp.Expression]:
    drange = _datetrunc_range(date, unit, dialect)
    if not drange:
        return None
    return exp.and_(left < date_literal(drange[0], target_type), left >= date_literal(drange[1], target_type), copy=False)