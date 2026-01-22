from __future__ import annotations
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from sqlglot import Dialect, expressions as exp
from sqlglot.helper import ensure_list
def _generate_edit_script(self, matching_set: t.Set[t.Tuple[int, int]]) -> t.List[Edit]:
    edit_script: t.List[Edit] = []
    for removed_node_id in self._unmatched_source_nodes:
        edit_script.append(Remove(self._source_index[removed_node_id]))
    for inserted_node_id in self._unmatched_target_nodes:
        edit_script.append(Insert(self._target_index[inserted_node_id]))
    for kept_source_node_id, kept_target_node_id in matching_set:
        source_node = self._source_index[kept_source_node_id]
        target_node = self._target_index[kept_target_node_id]
        if not isinstance(source_node, UPDATABLE_EXPRESSION_TYPES) or source_node == target_node:
            edit_script.extend(self._generate_move_edits(source_node, target_node, matching_set))
            edit_script.append(Keep(source_node, target_node))
        else:
            edit_script.append(Update(source_node, target_node))
    return edit_script