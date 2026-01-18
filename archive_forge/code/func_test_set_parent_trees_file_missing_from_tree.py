import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_set_parent_trees_file_missing_from_tree(self):
    tree1 = self.make_branch_and_memory_tree('tree1')
    tree1.lock_write()
    try:
        tree1.add('')
        tree1.add(['a file'], ['file'], [b'file-id'])
        tree1.put_file_bytes_non_atomic('a file', b'file-content')
        revid1 = tree1.commit('foo')
    finally:
        tree1.unlock()
    branch2 = tree1.branch.controldir.clone('tree2').open_branch()
    tree2 = memorytree.MemoryTree.create_on_branch(branch2)
    tree2.lock_write()
    try:
        tree2.put_file_bytes_non_atomic('a file', b'new file-content')
        revid2 = tree2.commit('foo')
        root_id = tree2.path2id('')
    finally:
        tree2.unlock()
    expected_result = ([revid1, revid2], [((b'', b'', root_id), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT), (b'd', b'', 0, False, revid1), (b'd', b'', 0, False, revid1)]), ((b'', b'a file', b'file-id'), [(b'a', b'', 0, False, b''), (b'f', b'2439573625385400f2a669657a7db6ae7515d371', 12, False, revid1), (b'f', b'542e57dc1cda4af37cb8e55ec07ce60364bb3c7d', 16, False, revid2)])])
    state = dirstate.DirState.initialize('dirstate')
    try:
        state.set_path_id(b'', root_id)
        state.set_parent_trees(((revid1, tree1.branch.repository.revision_tree(revid1)), (revid2, tree2.branch.repository.revision_tree(revid2))), [])
    except:
        state.unlock()
        raise
    else:
        self.check_state_with_reopen(expected_result, state)