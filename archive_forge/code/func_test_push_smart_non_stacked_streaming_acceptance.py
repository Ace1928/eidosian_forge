from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
def test_push_smart_non_stacked_streaming_acceptance(self):
    self.setup_smart_server_with_call_log()
    t = self.make_branch_and_tree('from')
    t.commit(allow_pointless=True, message='first commit')
    self.reset_smart_call_log()
    self.run_bzr(['push', self.get_url('to-one')], working_dir='from')
    self.assertLength(9, self.hpss_calls)
    self.assertLength(1, self.hpss_connections)
    self.assertThat(self.hpss_calls, ContainsNoVfsCalls)