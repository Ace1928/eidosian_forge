import time
from . import debug, errors, osutils, revision, trace
def _find_nodes_common_to_all_unique(self, unique_tip_searchers, all_unique_searcher, newly_seen_unique, step_all_unique):
    """Find nodes that are common to all unique_tip_searchers.

        If it is time, step the all_unique_searcher, and add its nodes to the
        result.
        """
    common_to_all_unique_nodes = newly_seen_unique.copy()
    for searcher in unique_tip_searchers:
        common_to_all_unique_nodes.intersection_update(searcher.seen)
    common_to_all_unique_nodes.intersection_update(all_unique_searcher.seen)
    if step_all_unique:
        tstart = osutils.perf_counter()
        nodes = all_unique_searcher.step()
        common_to_all_unique_nodes.update(nodes)
        if 'graph' in debug.debug_flags:
            tdelta = osutils.perf_counter() - tstart
            trace.mutter('all_unique_searcher step() took %.3fsfor %d nodes (%d total), iteration: %s', tdelta, len(nodes), len(all_unique_searcher.seen), all_unique_searcher._iterations)
    return common_to_all_unique_nodes