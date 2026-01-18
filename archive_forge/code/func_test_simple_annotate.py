from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
def test_simple_annotate(self):
    self.setup_smart_server_with_call_log()
    wt = self.make_branch_and_tree('branch')
    self.build_tree_contents([('branch/hello.txt', b'my helicopter\n')])
    wt.add(['hello.txt'])
    wt.commit('commit', committer='test@user')
    self.reset_smart_call_log()
    out, err = self.run_bzr(['annotate', '-d', self.get_url('branch'), 'hello.txt'])
    self.assertLength(9, self.hpss_calls)
    self.assertLength(1, self.hpss_connections)
    self.assertThat(self.hpss_calls, ContainsNoVfsCalls)