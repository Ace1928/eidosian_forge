import os
import tempfile
from breezy import osutils, tests, transport, urlutils
def test_url_log(self):
    url = self.get_readonly_url() + 'subdir/'
    out, err = self.run_bzr(['log', url], retcode=3)
    self.assertEqual('brz: ERROR: Not a branch: "%s".\n' % url, err)