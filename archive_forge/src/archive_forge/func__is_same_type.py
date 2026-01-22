from __future__ import annotations
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from sqlglot import Dialect, expressions as exp
from sqlglot.helper import ensure_list
def _is_same_type(source: exp.Expression, target: exp.Expression) -> bool:
    if type(source) is type(target):
        if isinstance(source, exp.Join):
            return source.args.get('side') == target.args.get('side')
        if isinstance(source, exp.Anonymous):
            return source.this == target.this
        return True
    return False