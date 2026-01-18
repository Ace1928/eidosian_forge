from io import StringIO
from .. import config
from .. import status as _mod_status
from ..revisionspec import RevisionSpec
from ..status import show_pending_merges, show_tree_status
from . import TestCaseWithTransport
def test_pending_with_ghosts(self):
    """Test when a pending merge's ancestry includes ghosts."""
    config.GlobalStack().set('email', 'Joe Foo <joe@foo.com>')
    tree = self.make_branch_and_tree('a')
    tree.commit('empty commit')
    tree2 = tree.controldir.clone('b').open_workingtree()
    tree2.commit('a non-ghost', timestamp=1196796819, timezone=0)
    tree2.add_parent_tree_id(b'a-ghost-revision')
    tree2.commit('commit with ghost', timestamp=1196796819, timezone=0)
    tree2.commit('another non-ghost', timestamp=1196796819, timezone=0)
    tree.merge_from_branch(tree2.branch)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    output = StringIO()
    show_pending_merges(tree, output, verbose=True)
    self.assertEqualDiff('pending merges:\n  Joe Foo 2007-12-04 another non-ghost\n    Joe Foo 2007-12-04 [merge] commit with ghost\n    (ghost) a-ghost-revision\n    Joe Foo 2007-12-04 a non-ghost\n', output.getvalue())