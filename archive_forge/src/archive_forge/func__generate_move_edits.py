from __future__ import annotations
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from sqlglot import Dialect, expressions as exp
from sqlglot.helper import ensure_list
def _generate_move_edits(self, source: exp.Expression, target: exp.Expression, matching_set: t.Set[t.Tuple[int, int]]) -> t.List[Move]:
    source_args = [id(e) for e in _expression_only_args(source)]
    target_args = [id(e) for e in _expression_only_args(target)]
    args_lcs = set(_lcs(source_args, target_args, lambda l, r: (l, r) in matching_set))
    move_edits = []
    for a in source_args:
        if a not in args_lcs and a not in self._unmatched_source_nodes:
            move_edits.append(Move(self._source_index[a]))
    return move_edits