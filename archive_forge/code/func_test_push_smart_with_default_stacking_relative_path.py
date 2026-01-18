import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_smart_with_default_stacking_relative_path(self):
    self.setup_smart_server_with_call_log()
    self.make_branch('stack-on', format='1.9')
    self.make_controldir('.').get_config().set_default_stack_on('stack-on')
    self.make_branch('from', format='1.9')
    out, err = self.run_bzr(['push', '-d', 'from', self.get_url('to')])
    b = branch.Branch.open(self.get_url('to'))
    self.assertEqual('../stack-on', b.get_stacked_on_url())