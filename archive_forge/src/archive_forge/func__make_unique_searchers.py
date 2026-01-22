import time
from . import debug, errors, osutils, revision, trace
def _make_unique_searchers(self, unique_nodes, unique_searcher, common_searcher):
    """Create a searcher for all the unique search tips (step 4).

        As a side effect, the common_searcher will stop searching any nodes
        that are ancestors of the unique searcher tips.

        :return: (all_unique_searcher, unique_tip_searchers)
        """
    unique_tips = self._remove_simple_descendants(unique_nodes, self.get_parent_map(unique_nodes))
    if len(unique_tips) == 1:
        unique_tip_searchers = []
        ancestor_all_unique = unique_searcher.find_seen_ancestors(unique_tips)
    else:
        unique_tip_searchers = []
        for tip in unique_tips:
            revs_to_search = unique_searcher.find_seen_ancestors([tip])
            revs_to_search.update(common_searcher.find_seen_ancestors(revs_to_search))
            searcher = self._make_breadth_first_searcher(revs_to_search)
            searcher._label = tip
            searcher.step()
            unique_tip_searchers.append(searcher)
        ancestor_all_unique = None
        for searcher in unique_tip_searchers:
            if ancestor_all_unique is None:
                ancestor_all_unique = set(searcher.seen)
            else:
                ancestor_all_unique = ancestor_all_unique.intersection(searcher.seen)
    all_unique_searcher = self._make_breadth_first_searcher(ancestor_all_unique)
    if ancestor_all_unique:
        all_unique_searcher.step()
        stopped_common = common_searcher.stop_searching_any(common_searcher.find_seen_ancestors(ancestor_all_unique))
        total_stopped = 0
        for searcher in unique_tip_searchers:
            total_stopped += len(searcher.stop_searching_any(searcher.find_seen_ancestors(ancestor_all_unique)))
    if 'graph' in debug.debug_flags:
        trace.mutter('For %d unique nodes, created %d + 1 unique searchers (%d stopped search tips, %d common ancestors (%d stopped common)', len(unique_nodes), len(unique_tip_searchers), total_stopped, len(ancestor_all_unique), len(stopped_common))
    return (all_unique_searcher, unique_tip_searchers)