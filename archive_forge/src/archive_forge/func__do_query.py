import time
from . import debug, errors, osutils, revision, trace
def _do_query(self, revisions):
    """Query for revisions.

        Adds revisions to the seen set.

        :param revisions: Revisions to query.
        :return: A tuple: (set(found_revisions), set(ghost_revisions),
           set(parents_of_found_revisions), dict(found_revisions:parents)).
        """
    found_revisions = set()
    parents_of_found = set()
    seen = self.seen
    seen.update(revisions)
    parent_map = self._parents_provider.get_parent_map(revisions)
    found_revisions.update(parent_map)
    for rev_id, parents in parent_map.items():
        if parents is None:
            continue
        new_found_parents = [p for p in parents if p not in seen]
        if new_found_parents:
            parents_of_found.update(new_found_parents)
    ghost_revisions = revisions - found_revisions
    return (found_revisions, ghost_revisions, parents_of_found, parent_map)