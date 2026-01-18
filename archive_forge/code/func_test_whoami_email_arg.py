from breezy import branch, config, errors, tests
from ..test_bedding import override_whoami
def test_whoami_email_arg(self):
    out = self.run_bzr("whoami --email 'foo <foo@example.com>'", 3)[0]
    self.assertEqual('', out)