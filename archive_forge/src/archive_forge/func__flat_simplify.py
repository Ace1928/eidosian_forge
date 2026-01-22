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
def _flat_simplify(expression, simplifier, root=True):
    if root or not expression.same_parent:
        operands = []
        queue = deque(expression.flatten(unnest=False))
        size = len(queue)
        while queue:
            a = queue.popleft()
            for b in queue:
                result = simplifier(expression, a, b)
                if result and result is not expression:
                    queue.remove(b)
                    queue.appendleft(result)
                    break
            else:
                operands.append(a)
        if len(operands) < size:
            return functools.reduce(lambda a, b: expression.__class__(this=a, expression=b), operands)
    return expression