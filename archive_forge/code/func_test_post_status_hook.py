from io import StringIO
from .. import config
from .. import status as _mod_status
from ..revisionspec import RevisionSpec
from ..status import show_pending_merges, show_tree_status
from . import TestCaseWithTransport
def test_post_status_hook(self):
    """Ensure that post_status hook is invoked with the right args.
        """
    calls = []
    _mod_status.hooks.install_named_hook('post_status', calls.append, None)
    self.assertLength(0, calls)
    tree = self.make_branch_and_tree('.')
    r1_id = tree.commit('one', allow_pointless=True)
    r2_id = tree.commit('two', allow_pointless=True)
    output = StringIO()
    show_tree_status(tree, to_file=output, revision=[RevisionSpec.from_string('revid:%s' % r1_id.decode('utf-8')), RevisionSpec.from_string('revid:%s' % r2_id.decode('utf-8'))])
    self.assertLength(1, calls)
    params = calls[0]
    self.assertIsInstance(params, _mod_status.StatusHookParams)
    attrs = ['old_tree', 'new_tree', 'to_file', 'versioned', 'show_ids', 'short', 'verbose', 'specific_files']
    for a in attrs:
        self.assertTrue(hasattr(params, a), 'Attribute "%s" not found in StatusHookParam' % a)