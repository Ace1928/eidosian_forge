from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
class TestSmartServerPush(TestCaseWithTransport):

    def test_push_smart_non_stacked_streaming_acceptance(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('from')
        t.commit(allow_pointless=True, message='first commit')
        self.reset_smart_call_log()
        self.run_bzr(['push', self.get_url('to-one')], working_dir='from')
        self.assertLength(9, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)

    def test_push_smart_stacked_streaming_acceptance(self):
        self.setup_smart_server_with_call_log()
        parent = self.make_branch_and_tree('parent', format='1.9')
        parent.commit(message='first commit')
        local = parent.controldir.sprout('local').open_workingtree()
        local.commit(message='local commit')
        self.reset_smart_call_log()
        self.run_bzr(['push', '--stacked', '--stacked-on', '../parent', self.get_url('public')], working_dir='local')
        self.assertLength(15, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)
        remote = branch.Branch.open('public')
        self.assertEndsWith(remote.get_stacked_on_url(), '/parent')

    def test_push_smart_tags_streaming_acceptance(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('from')
        rev_id = t.commit(allow_pointless=True, message='first commit')
        t.branch.tags.set_tag('new-tag', rev_id)
        self.reset_smart_call_log()
        self.run_bzr(['push', self.get_url('to-one')], working_dir='from')
        self.assertLength(11, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)

    def test_push_smart_incremental_acceptance(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('from')
        rev_id1 = t.commit(allow_pointless=True, message='first commit')
        rev_id2 = t.commit(allow_pointless=True, message='second commit')
        self.run_bzr(['push', self.get_url('to-one'), '-r1'], working_dir='from')
        self.reset_smart_call_log()
        self.run_bzr(['push', self.get_url('to-one')], working_dir='from')
        self.assertLength(11, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)