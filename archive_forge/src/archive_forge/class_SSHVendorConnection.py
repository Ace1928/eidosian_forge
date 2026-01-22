import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
class SSHVendorConnection(TestCaseWithSFTPServer):
    """Test that the ssh vendors can all connect.

    Verify that a full-handshake (SSH over loopback TCP) sftp connection works.

    We have 3 sftp implementations in the test suite:
      'loopback': Doesn't use ssh, just uses a local socket. Most tests are
                  done this way to save the handshaking time, so it is not
                  tested again here
      'none':     This uses paramiko's built-in ssh client and server, and
                  layers sftp on top of it.
      None:       If 'ssh' exists on the machine, then it will be spawned as a
                  child process.
    """

    def setUp(self):
        super().setUp()

        def create_server():
            """Just a wrapper so that when created, it will set _vendor"""
            server = stub_sftp.SFTPFullAbsoluteServer()
            server._vendor = self._test_vendor
            return server
        self._test_vendor = 'loopback'
        self.vfs_transport_server = create_server
        f = open('a_file', 'wb')
        try:
            f.write(b'foobar\n')
        finally:
            f.close()

    def set_vendor(self, vendor):
        self._test_vendor = vendor

    def test_connection_paramiko(self):
        from breezy.transport import ssh
        self.set_vendor(ssh.ParamikoVendor())
        t = self.get_transport()
        self.assertEqual(b'foobar\n', t.get('a_file').read())

    def test_connection_vendor(self):
        raise TestSkipped("We don't test spawning real ssh, because it prompts for a password. Enable this test if we figure out how to prevent this.")
        self.set_vendor(None)
        t = self.get_transport()
        self.assertEqual(b'foobar\n', t.get('a_file').read())