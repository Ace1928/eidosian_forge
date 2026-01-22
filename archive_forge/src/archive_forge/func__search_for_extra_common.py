import time
from . import debug, errors, osutils, revision, trace
def _search_for_extra_common(self, common, searchers):
    """Make sure that unique nodes are genuinely unique.

        After _find_border_ancestors, all nodes marked "common" are indeed
        common. Some of the nodes considered unique are not, due to history
        shortcuts stopping the searches early.

        We know that we have searched enough when all common search tips are
        descended from all unique (uncommon) nodes because we know that a node
        cannot be an ancestor of its own ancestor.

        :param common: A set of common nodes
        :param searchers: The searchers returned from _find_border_ancestors
        :return: None
        """
    if len(searchers) != 2:
        raise NotImplementedError('Algorithm not yet implemented for > 2 searchers')
    common_searchers = searchers
    left_searcher = searchers[0]
    right_searcher = searchers[1]
    unique = left_searcher.seen.symmetric_difference(right_searcher.seen)
    if not unique:
        return
    total_unique = len(unique)
    unique = self._remove_simple_descendants(unique, self.get_parent_map(unique))
    simple_unique = len(unique)
    unique_searchers = []
    for revision_id in unique:
        if revision_id in left_searcher.seen:
            parent_searcher = left_searcher
        else:
            parent_searcher = right_searcher
        revs_to_search = parent_searcher.find_seen_ancestors([revision_id])
        if not revs_to_search:
            revs_to_search = [revision_id]
        searcher = self._make_breadth_first_searcher(revs_to_search)
        searcher.step()
        unique_searchers.append(searcher)
    ancestor_all_unique = None
    for searcher in unique_searchers:
        if ancestor_all_unique is None:
            ancestor_all_unique = set(searcher.seen)
        else:
            ancestor_all_unique = ancestor_all_unique.intersection(searcher.seen)
    trace.mutter('Started %d unique searchers for %d unique revisions', simple_unique, total_unique)
    while True:
        newly_seen_common = set()
        for searcher in common_searchers:
            newly_seen_common.update(searcher.step())
        newly_seen_unique = set()
        for searcher in unique_searchers:
            newly_seen_unique.update(searcher.step())
        new_common_unique = set()
        for revision in newly_seen_unique:
            for searcher in unique_searchers:
                if revision not in searcher.seen:
                    break
            else:
                new_common_unique.add(revision)
        if newly_seen_common:
            for searcher in common_searchers:
                newly_seen_common.update(searcher.find_seen_ancestors(newly_seen_common))
            for searcher in common_searchers:
                searcher.start_searching(newly_seen_common)
            stop_searching_common = ancestor_all_unique.intersection(newly_seen_common)
            if stop_searching_common:
                for searcher in common_searchers:
                    searcher.stop_searching_any(stop_searching_common)
        if new_common_unique:
            for searcher in unique_searchers:
                new_common_unique.update(searcher.find_seen_ancestors(new_common_unique))
            for searcher in common_searchers:
                new_common_unique.update(searcher.find_seen_ancestors(new_common_unique))
            for searcher in unique_searchers:
                searcher.start_searching(new_common_unique)
            for searcher in common_searchers:
                searcher.stop_searching_any(new_common_unique)
            ancestor_all_unique.update(new_common_unique)
            next_unique_searchers = []
            unique_search_sets = set()
            for searcher in unique_searchers:
                will_search_set = frozenset(searcher._next_query)
                if will_search_set not in unique_search_sets:
                    unique_search_sets.add(will_search_set)
                    next_unique_searchers.append(searcher)
            unique_searchers = next_unique_searchers
        for searcher in common_searchers:
            if searcher._next_query:
                break
        else:
            return