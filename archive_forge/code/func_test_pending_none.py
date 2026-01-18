from io import StringIO
from .. import config
from .. import status as _mod_status
from ..revisionspec import RevisionSpec
from ..status import show_pending_merges, show_tree_status
from . import TestCaseWithTransport
def test_pending_none(self):
    tree = self.make_branch_and_tree('a')
    tree.commit('empty commit')
    tree2 = self.make_branch_and_tree('b')
    tree2.add_parent_tree_id(b'some-ghost', allow_leftmost_as_ghost=True)
    tree2.merge_from_branch(tree.branch)
    output = StringIO()
    with tree2.lock_read():
        show_pending_merges(tree2, output)
    self.assertContainsRe(output.getvalue(), 'empty commit')