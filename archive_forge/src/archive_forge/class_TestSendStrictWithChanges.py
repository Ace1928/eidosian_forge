from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
class TestSendStrictWithChanges(tests.TestCaseWithTransport, TestSendStrictMixin):
    scenarios = [('uncommitted', dict(_changes_type='_uncommitted_changes')), ('pending_merges', dict(_changes_type='_pending_merges')), ('out-of-sync-trees', dict(_changes_type='_out_of_sync_trees'))]
    _changes_type = None

    def setUp(self):
        super().setUp()
        do_changes_func = getattr(self, self._changes_type)
        do_changes_func()

    def _uncommitted_changes(self):
        self.parent, self.local = self.make_parent_and_local_branches()
        self.build_tree_contents([('local/file', b'modified')])

    def _pending_merges(self):
        self.parent, self.local = self.make_parent_and_local_branches()
        other_bzrdir = self.parent_tree.controldir.sprout('other')
        other_tree = other_bzrdir.open_workingtree()
        self.build_tree_contents([('other/other-file', b'other')])
        other_tree.add('other-file')
        other_tree.commit('other commit', rev_id=b'other')
        self.local_tree.merge_from_branch(other_tree.branch)
        self.local_tree.revert(filenames=['other-file'], backups=False)

    def _out_of_sync_trees(self):
        self.parent, self.local = self.make_parent_and_local_branches()
        self.run_bzr(['checkout', '--lightweight', 'local', 'checkout'])
        self.build_tree_contents([('local/file', b'modified in local')])
        self.local_tree.commit('modify file', rev_id=b'modified-in-local')
        self._default_wd = 'checkout'
        self._default_errors = ["Working tree is out of date, please run 'brz update'\\."]
        self._default_sent_revs = [b'modified-in-local', self.local]

    def test_send_default(self):
        self.assertSendSucceeds([], with_warning=True)

    def test_send_with_revision(self):
        self.assertSendSucceeds(['-r', 'revid:' + self.local.decode('utf-8')], revs=[self.local])

    def test_send_no_strict(self):
        self.assertSendSucceeds(['--no-strict'])

    def test_send_strict_with_changes(self):
        self.assertSendFails(['--strict'])

    def test_send_respect_config_var_strict(self):
        self.set_config_send_strict('true')
        self.assertSendFails([])
        self.assertSendSucceeds(['--no-strict'])

    def test_send_bogus_config_var_ignored(self):
        self.set_config_send_strict("I'm unsure")
        self.assertSendSucceeds([], with_warning=True)

    def test_send_no_strict_command_line_override_config(self):
        self.set_config_send_strict('true')
        self.assertSendFails([])
        self.assertSendSucceeds(['--no-strict'])

    def test_send_strict_command_line_override_config(self):
        self.set_config_send_strict('false')
        self.assertSendSucceeds([])
        self.assertSendFails(['--strict'])