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
def uniq_sort(expression, root=True):
    """
    Uniq and sort a connector.

    C AND A AND B AND B -> A AND B AND C
    """
    if isinstance(expression, exp.Connector) and (root or not expression.same_parent):
        result_func = exp.and_ if isinstance(expression, exp.And) else exp.or_
        flattened = tuple(expression.flatten())
        deduped = {gen(e): e for e in flattened}
        arr = tuple(deduped.items())
        for i, (sql, e) in enumerate(arr[1:]):
            if sql < arr[i][0]:
                expression = result_func(*(e for _, e in sorted(arr)), copy=False)
                break
        else:
            if len(deduped) < len(flattened):
                expression = result_func(*deduped.values(), copy=False)
    return expression