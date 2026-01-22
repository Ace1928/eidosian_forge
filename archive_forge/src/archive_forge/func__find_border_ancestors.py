import time
from . import debug, errors, osutils, revision, trace
def _find_border_ancestors(self, revisions):
    """Find common ancestors with at least one uncommon descendant.

        Border ancestors are identified using a breadth-first
        search starting at the bottom of the graph.  Searches are stopped
        whenever a node or one of its descendants is determined to be common.

        This will scale with the number of uncommon ancestors.

        As well as the border ancestors, a set of seen common ancestors and a
        list of sets of seen ancestors for each input revision is returned.
        This allows calculation of graph difference from the results of this
        operation.
        """
    if None in revisions:
        raise errors.InvalidRevisionId(None, self)
    common_ancestors = set()
    searchers = [self._make_breadth_first_searcher([r]) for r in revisions]
    border_ancestors = set()
    while True:
        newly_seen = set()
        for searcher in searchers:
            new_ancestors = searcher.step()
            if new_ancestors:
                newly_seen.update(new_ancestors)
        new_common = set()
        for revision in newly_seen:
            if revision in common_ancestors:
                new_common.add(revision)
                continue
            for searcher in searchers:
                if revision not in searcher.seen:
                    break
            else:
                border_ancestors.add(revision)
                new_common.add(revision)
        if new_common:
            for searcher in searchers:
                new_common.update(searcher.find_seen_ancestors(new_common))
            for searcher in searchers:
                searcher.start_searching(new_common)
            common_ancestors.update(new_common)
        unique_search_sets = set()
        for searcher in searchers:
            will_search_set = frozenset(searcher._next_query)
            if will_search_set not in unique_search_sets:
                unique_search_sets.add(will_search_set)
        if len(unique_search_sets) == 1:
            nodes = unique_search_sets.pop()
            uncommon_nodes = nodes.difference(common_ancestors)
            if uncommon_nodes:
                raise AssertionError('Somehow we ended up converging without actually marking them as in common.\nStart_nodes: %s\nuncommon_nodes: %s' % (revisions, uncommon_nodes))
            break
    return (border_ancestors, common_ancestors, searchers)