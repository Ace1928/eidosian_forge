from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
def test_sign_single_commit(self):
    self.setup_smart_server_with_call_log()
    t = self.make_branch_and_tree('branch')
    self.build_tree_contents([('branch/foo', b'thecontents')])
    t.add('foo')
    t.commit('message')
    self.reset_smart_call_log()
    self.monkey_patch_gpg()
    out, err = self.run_bzr(['sign-my-commits', self.get_url('branch')])
    self.assertLength(15, self.hpss_calls)
    self.assertLength(1, self.hpss_connections)
    self.assertThat(self.hpss_calls, ContainsNoVfsCalls)