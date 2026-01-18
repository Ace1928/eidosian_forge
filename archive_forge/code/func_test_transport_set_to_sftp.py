import os
from breezy import tests
from breezy.tests import features
from breezy.transport import memory
def test_transport_set_to_sftp(self):
    self.requireFeature(features.paramiko)
    from breezy.tests import stub_sftp
    params = self.get_params_passed_to_core('selftest --transport=sftp')
    self.assertEqual(stub_sftp.SFTPAbsoluteServer, params[1]['transport'])