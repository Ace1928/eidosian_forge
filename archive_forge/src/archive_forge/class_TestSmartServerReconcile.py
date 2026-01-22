from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
class TestSmartServerReconcile(TestCaseWithTransport):

    def test_simple_reconcile(self):
        self.setup_smart_server_with_call_log()
        self.make_branch('branch')
        self.reset_smart_call_log()
        out, err = self.run_bzr(['reconcile', self.get_url('branch')])
        self.assertLength(10, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)