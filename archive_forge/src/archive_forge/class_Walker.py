import collections
import heapq
from itertools import chain
from typing import Deque, Dict, List, Optional, Set, Tuple
from .diff_tree import (
from .errors import MissingCommitError
from .objects import Commit, ObjectID, Tag
class Walker:
    """Object for performing a walk of commits in a store.

    Walker objects are initialized with a store and other options and can then
    be treated as iterators of Commit objects.
    """

    def __init__(self, store, include: List[bytes], exclude: Optional[List[bytes]]=None, order: str='date', reverse: bool=False, max_entries: Optional[int]=None, paths: Optional[List[bytes]]=None, rename_detector: Optional[RenameDetector]=None, follow: bool=False, since: Optional[int]=None, until: Optional[int]=None, get_parents=lambda commit: commit.parents, queue_cls=_CommitTimeQueue) -> None:
        """Constructor.

        Args:
          store: ObjectStore instance for looking up objects.
          include: Iterable of SHAs of commits to include along with their
            ancestors.
          exclude: Iterable of SHAs of commits to exclude along with their
            ancestors, overriding includes.
          order: ORDER_* constant specifying the order of results.
            Anything other than ORDER_DATE may result in O(n) memory usage.
          reverse: If True, reverse the order of output, requiring O(n)
            memory.
          max_entries: The maximum number of entries to yield, or None for
            no limit.
          paths: Iterable of file or subtree paths to show entries for.
          rename_detector: diff.RenameDetector object for detecting
            renames.
          follow: If True, follow path across renames/copies. Forces a
            default rename_detector.
          since: Timestamp to list commits after.
          until: Timestamp to list commits before.
          get_parents: Method to retrieve the parents of a commit
          queue_cls: A class to use for a queue of commits, supporting the
            iterator protocol. The constructor takes a single argument, the
            Walker.
        """
        if order not in ALL_ORDERS:
            raise ValueError('Unknown walk order %s' % order)
        self.store = store
        if isinstance(include, bytes):
            include = [include]
        self.include = include
        self.excluded = set(exclude or [])
        self.order = order
        self.reverse = reverse
        self.max_entries = max_entries
        self.paths = paths and set(paths) or None
        if follow and (not rename_detector):
            rename_detector = RenameDetector(store)
        self.rename_detector = rename_detector
        self.get_parents = get_parents
        self.follow = follow
        self.since = since
        self.until = until
        self._num_entries = 0
        self._queue = queue_cls(self)
        self._out_queue: Deque[WalkEntry] = collections.deque()

    def _path_matches(self, changed_path):
        if changed_path is None:
            return False
        if self.paths is None:
            return True
        for followed_path in self.paths:
            if changed_path == followed_path:
                return True
            if changed_path.startswith(followed_path) and changed_path[len(followed_path)] == b'/'[0]:
                return True
        return False

    def _change_matches(self, change):
        if not change:
            return False
        old_path = change.old.path
        new_path = change.new.path
        if self._path_matches(new_path):
            if self.follow and change.type in RENAME_CHANGE_TYPES:
                self.paths.add(old_path)
                self.paths.remove(new_path)
            return True
        elif self._path_matches(old_path):
            return True
        return False

    def _should_return(self, entry):
        """Determine if a walk entry should be returned..

        Args:
          entry: The WalkEntry to consider.
        Returns: True if the WalkEntry should be returned by this walk, or
            False otherwise (e.g. if it doesn't match any requested paths).
        """
        commit = entry.commit
        if self.since is not None and commit.commit_time < self.since:
            return False
        if self.until is not None and commit.commit_time > self.until:
            return False
        if commit.id in self.excluded:
            return False
        if self.paths is None:
            return True
        if len(self.get_parents(commit)) > 1:
            for path_changes in entry.changes():
                for change in path_changes:
                    if self._change_matches(change):
                        return True
        else:
            for change in entry.changes():
                if self._change_matches(change):
                    return True
        return None

    def _next(self):
        max_entries = self.max_entries
        while max_entries is None or self._num_entries < max_entries:
            entry = next(self._queue)
            if entry is not None:
                self._out_queue.append(entry)
            if entry is None or len(self._out_queue) > _MAX_EXTRA_COMMITS:
                if not self._out_queue:
                    return None
                entry = self._out_queue.popleft()
                if self._should_return(entry):
                    self._num_entries += 1
                    return entry
        return None

    def _reorder(self, results):
        """Possibly reorder a results iterator.

        Args:
          results: An iterator of WalkEntry objects, in the order returned
            from the queue_cls.
        Returns: An iterator or list of WalkEntry objects, in the order
            required by the Walker.
        """
        if self.order == ORDER_TOPO:
            results = _topo_reorder(results, self.get_parents)
        if self.reverse:
            results = reversed(list(results))
        return results

    def __iter__(self):
        return iter(self._reorder(iter(self._next, None)))