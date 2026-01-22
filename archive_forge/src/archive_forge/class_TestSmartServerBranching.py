from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
class TestSmartServerBranching(TestCaseWithTransport):

    def test_branch_from_trivial_branch_to_same_server_branch_acceptance(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('from')
        for count in range(9):
            t.commit(message='commit %d' % count)
        self.reset_smart_call_log()
        out, err = self.run_bzr(['branch', self.get_url('from'), self.get_url('target')])
        self.assertLength(2, self.hpss_connections)
        self.assertLength(34, self.hpss_calls)
        self.expectFailure('branching to the same branch requires VFS access', self.assertThat, self.hpss_calls, ContainsNoVfsCalls)

    def test_branch_from_trivial_branch_streaming_acceptance(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('from')
        for count in range(9):
            t.commit(message='commit %d' % count)
        self.reset_smart_call_log()
        out, err = self.run_bzr(['branch', self.get_url('from'), 'local-target'])
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)
        self.assertLength(11, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)

    def test_branch_from_trivial_stacked_branch_streaming_acceptance(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('trunk')
        for count in range(8):
            t.commit(message='commit %d' % count)
        tree2 = t.branch.controldir.sprout('feature', stacked=True).open_workingtree()
        local_tree = t.branch.controldir.sprout('local-working').open_workingtree()
        local_tree.commit('feature change')
        local_tree.branch.push(tree2.branch)
        self.reset_smart_call_log()
        out, err = self.run_bzr(['branch', self.get_url('feature'), 'local-target'])
        self.assertLength(16, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)

    def test_branch_from_branch_with_tags(self):
        self.setup_smart_server_with_call_log()
        builder = self.make_branch_builder('source')
        source, rev1, rev2 = fixtures.build_branch_with_non_ancestral_rev(builder)
        source.get_config_stack().set('branch.fetch_tags', True)
        source.tags.set_tag('tag-a', rev2)
        source.tags.set_tag('tag-missing', b'missing-rev')
        self.reset_smart_call_log()
        out, err = self.run_bzr(['branch', self.get_url('source'), 'target'])
        self.assertLength(11, self.hpss_calls)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)
        self.assertLength(1, self.hpss_connections)

    def test_branch_to_stacked_from_trivial_branch_streaming_acceptance(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('from')
        for count in range(9):
            t.commit(message='commit %d' % count)
        self.reset_smart_call_log()
        out, err = self.run_bzr(['branch', '--stacked', self.get_url('from'), 'local-target'])
        readvs_of_rix_files = [c for c in self.hpss_calls if c.call.method == 'readv' and c.call.args[-1].endswith('.rix')]
        self.assertLength(1, self.hpss_connections)
        self.assertLength(0, readvs_of_rix_files)
        self.expectFailure('branching to stacked requires VFS access', self.assertThat, self.hpss_calls, ContainsNoVfsCalls)

    def test_branch_from_branch_with_ghosts(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('from')
        for count in range(9):
            t.commit(message='commit %d' % count)
        t.set_parent_ids([t.last_revision(), b'ghost'])
        t.commit(message='add commit with parent')
        self.reset_smart_call_log()
        out, err = self.run_bzr(['branch', self.get_url('from'), 'local-target'])
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)
        self.assertLength(12, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)