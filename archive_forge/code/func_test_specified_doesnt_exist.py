import os
import sys
from .. import config, tests, trace
from ..transport.http import opt_ssl_ca_certs, ssl
def test_specified_doesnt_exist(self):
    stack = self.get_stack('')
    self.overrideAttr(opt_ssl_ca_certs, 'default', os.path.join(self.test_dir, 'nonexisting.pem'))
    self.warnings = []

    def warning(*args):
        self.warnings.append(args[0] % args[1:])
    self.overrideAttr(trace, 'warning', warning)
    self.assertEqual(None, stack.get('ssl.ca_certs'))
    self.assertLength(1, self.warnings)
    self.assertContainsRe(self.warnings[0], 'is not valid for "ssl.ca_certs"')