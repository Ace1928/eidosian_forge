import itertools
from .. import debug, revision, trace
from ..graph import DictParentsProvider, Graph, invert_parent_map
from ..repository import AbstractSearchResult
def _run_search(parent_map, heads, exclude_keys):
    """Given a parent map, run a _BreadthFirstSearcher on it.

    Start at heads, walk until you hit exclude_keys. As a further improvement,
    watch for any heads that you encounter while walking, which means they were
    not heads of the search.

    This is mostly used to generate a succinct recipe for how to walk through
    most of parent_map.

    :return: (_BreadthFirstSearcher, set(heads_encountered_by_walking))
    """
    g = Graph(DictParentsProvider(parent_map))
    s = g._make_breadth_first_searcher(heads)
    found_heads = set()
    while True:
        try:
            next_revs = next(s)
        except StopIteration:
            break
        for parents in s._current_parents.values():
            f_heads = heads.intersection(parents)
            if f_heads:
                found_heads.update(f_heads)
        stop_keys = exclude_keys.intersection(next_revs)
        if stop_keys:
            s.stop_searching_any(stop_keys)
    for parents in s._current_parents.values():
        f_heads = heads.intersection(parents)
        if f_heads:
            found_heads.update(f_heads)
    return (s, found_heads)