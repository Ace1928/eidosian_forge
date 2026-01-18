import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_set_parent_trees_no_content(self):
    tree1 = self.make_branch_and_memory_tree('tree1')
    tree1.lock_write()
    try:
        tree1.add('')
        revid1 = tree1.commit('foo')
    finally:
        tree1.unlock()
    branch2 = tree1.branch.controldir.clone('tree2').open_branch()
    tree2 = memorytree.MemoryTree.create_on_branch(branch2)
    tree2.lock_write()
    try:
        revid2 = tree2.commit('foo')
        root_id = tree2.path2id('')
    finally:
        tree2.unlock()
    state = dirstate.DirState.initialize('dirstate')
    try:
        state.set_path_id(b'', root_id)
        state.set_parent_trees(((revid1, tree1.branch.repository.revision_tree(revid1)), (revid2, tree2.branch.repository.revision_tree(revid2)), (b'ghost-rev', None)), [b'ghost-rev'])
        state._validate()
        state.save()
        state._validate()
    finally:
        state.unlock()
    state = dirstate.DirState.on_file('dirstate')
    state.lock_write()
    try:
        self.assertEqual([revid1, revid2, b'ghost-rev'], state.get_parent_ids())
        list(state._iter_entries())
        state.set_parent_trees(((revid1, tree1.branch.repository.revision_tree(revid1)), (b'ghost-rev', None)), [b'ghost-rev'])
        state.set_parent_trees(((revid1, tree1.branch.repository.revision_tree(revid1)), (revid2, tree2.branch.repository.revision_tree(revid2)), (b'ghost-rev', tree2.branch.repository.revision_tree(_mod_revision.NULL_REVISION))), [b'ghost-rev'])
        self.assertEqual([revid1, revid2, b'ghost-rev'], state.get_parent_ids())
        self.assertEqual([b'ghost-rev'], state.get_ghosts())
        self.assertEqual([((b'', b'', root_id), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT), (b'd', b'', 0, False, revid1), (b'd', b'', 0, False, revid1)])], list(state._iter_entries()))
    finally:
        state.unlock()