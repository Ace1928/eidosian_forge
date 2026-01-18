import os
import sys
from breezy import urlutils
from breezy.branch import Branch
from breezy.controldir import ControlDir
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_sftp_server_modes(self):
    if sys.platform == 'win32':
        raise TestSkipped('chmod has no effect on win32')
    umask = 18
    original_umask = os.umask(umask)
    try:
        t = self.get_transport()
        with t._sftp_open_exclusive('a', mode=438) as f:
            f.write(b'foo\n')
        self.assertTransportMode(t, 'a', 438 & ~umask)
        t.put_bytes('b', b'txt', mode=438)
        self.assertTransportMode(t, 'b', 438)
        t._get_sftp().mkdir('c', mode=511)
        self.assertTransportMode(t, 'c', 511 & ~umask)
        t.mkdir('d', mode=511)
        self.assertTransportMode(t, 'd', 511)
    finally:
        os.umask(original_umask)