from io import StringIO
from .. import config
from .. import status as _mod_status
from ..revisionspec import RevisionSpec
from ..status import show_pending_merges, show_tree_status
from . import TestCaseWithTransport
def test_with_pending_ghost(self):
    """Test when a pending merge is itself a ghost"""
    tree = self.make_branch_and_tree('a')
    tree.commit('first')
    tree.add_parent_tree_id(b'a-ghost-revision')
    tree.lock_read()
    self.addCleanup(tree.unlock)
    output = StringIO()
    show_pending_merges(tree, output)
    self.assertEqualDiff('pending merge tips: (use -v to see all merge revisions)\n  (ghost) a-ghost-revision\n', output.getvalue())