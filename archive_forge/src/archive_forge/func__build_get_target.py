from __future__ import annotations
import copy
from collections import defaultdict, deque, namedtuple
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Callable, Literal, NamedTuple, overload
from dask.core import get_dependencies, get_deps, getcycle, istask, reverse_dict
from dask.typing import Key
def _build_get_target() -> Callable[[], Key]:
    occurrences: defaultdict[Key, int] = defaultdict(int)
    for t in leaf_nodes:
        for r in roots_connected[t]:
            occurrences[r] += 1
    occurences_grouped = defaultdict(set)
    for root, occ in occurrences.items():
        occurences_grouped[occ].add(root)
    occurences_grouped_sorted = {}
    for k, v in occurences_grouped.items():
        occurences_grouped_sorted[k] = sorted(v, key=sort_key, reverse=True)
    del occurences_grouped, occurrences

    def pick_seed() -> Key | None:
        while occurences_grouped_sorted:
            key = max(occurences_grouped_sorted)
            picked_root = occurences_grouped_sorted[key][-1]
            if picked_root in result:
                occurences_grouped_sorted[key].pop()
                if not occurences_grouped_sorted[key]:
                    del occurences_grouped_sorted[key]
                continue
            return picked_root
        return None

    def get_target() -> Key:
        candidates = leaf_nodes
        skey: Callable = sort_key
        if runnable_hull:
            skey = lambda k: (num_needed[k], sort_key(k))
            candidates = runnable_hull & candidates
        elif reachable_hull:
            skey = lambda k: (num_needed[k], sort_key(k))
            candidates = reachable_hull & candidates
        if not candidates:
            if (seed := pick_seed()):
                candidates = leafs_connected[seed]
            else:
                candidates = runnable_hull or reachable_hull
        return min(candidates, key=skey)
    return get_target