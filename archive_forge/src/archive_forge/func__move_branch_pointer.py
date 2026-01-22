from . import commit, controldir, errors, revision
def _move_branch_pointer(self, new_revision_id, allow_leftmost_as_ghost=False):
    """Point self._branch to a different revision id."""
    with self._branch.lock_write():
        cur_revno, cur_revision_id = self._branch.last_revision_info()
        try:
            g = self._branch.repository.get_graph()
            new_revno = g.find_distance_to_null(new_revision_id, [(cur_revision_id, cur_revno)])
            self._branch.set_last_revision_info(new_revno, new_revision_id)
        except errors.GhostRevisionsHaveNoRevno:
            if not allow_leftmost_as_ghost:
                raise
            new_revno = 1
    if self._tree is not None:
        new_tree = self._branch.create_memorytree()
        new_tree.lock_write()
        self._tree.unlock()
        self._tree = new_tree