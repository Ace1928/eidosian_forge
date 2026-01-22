from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
class TestSmartServerSend(TestCaseWithTransport):

    def test_send(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/foo', b'thecontents')])
        t.add('foo')
        t.commit('message')
        local = t.controldir.sprout('local-branch').open_workingtree()
        self.build_tree_contents([('branch/foo', b'thenewcontents')])
        local.commit('anothermessage')
        self.reset_smart_call_log()
        out, err = self.run_bzr(['send', '-o', 'x.diff', self.get_url('branch')], working_dir='local-branch')
        self.assertLength(7, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)