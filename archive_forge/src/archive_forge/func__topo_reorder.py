import collections
import heapq
from itertools import chain
from typing import Deque, Dict, List, Optional, Set, Tuple
from .diff_tree import (
from .errors import MissingCommitError
from .objects import Commit, ObjectID, Tag
def _topo_reorder(entries, get_parents=lambda commit: commit.parents):
    """Reorder an iterable of entries topologically.

    This works best assuming the entries are already in almost-topological
    order, e.g. in commit time order.

    Args:
      entries: An iterable of WalkEntry objects.
      get_parents: Optional function for getting the parents of a commit.
    Returns: iterator over WalkEntry objects from entries in FIFO order, except
        where a parent would be yielded before any of its children.
    """
    todo = collections.deque()
    pending = {}
    num_children = collections.defaultdict(int)
    for entry in entries:
        todo.append(entry)
        for p in get_parents(entry.commit):
            num_children[p] += 1
    while todo:
        entry = todo.popleft()
        commit = entry.commit
        commit_id = commit.id
        if num_children[commit_id]:
            pending[commit_id] = entry
            continue
        for parent_id in get_parents(commit):
            num_children[parent_id] -= 1
            if not num_children[parent_id]:
                parent_entry = pending.pop(parent_id, None)
                if parent_entry:
                    todo.appendleft(parent_entry)
        yield entry