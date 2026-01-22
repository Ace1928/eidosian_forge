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
class SFTPNonServerTest(TestCase):

    def setUp(self):
        super().setUp()
        self.requireFeature(features.paramiko)

    def test_parse_url_with_home_dir(self):
        s = _mod_sftp.SFTPTransport('sftp://ro%62ey:h%40t@example.com:2222/~/relative')
        self.assertEqual(s._parsed_url.host, 'example.com')
        self.assertEqual(s._parsed_url.port, 2222)
        self.assertEqual(s._parsed_url.user, 'robey')
        self.assertEqual(s._parsed_url.password, 'h@t')
        self.assertEqual(s._parsed_url.path, '/~/relative/')

    def test_relpath(self):
        s = _mod_sftp.SFTPTransport('sftp://user@host.com/abs/path')
        self.assertRaises(errors.PathNotChild, s.relpath, 'sftp://user@host.com/~/rel/path/sub')

    def test_get_paramiko_vendor(self):
        """Test that if no 'ssh' is available we get builtin paramiko"""
        from breezy.transport import ssh
        self.overrideAttr(ssh, '_ssh_vendor_manager')
        self.overrideEnv('PATH', '.')
        ssh._ssh_vendor_manager.clear_cache()
        vendor = ssh._get_ssh_vendor()
        self.assertIsInstance(vendor, ssh.ParamikoVendor)

    def test_abspath_root_sibling_server(self):
        server = stub_sftp.SFTPSiblingAbsoluteServer()
        server.start_server()
        self.addCleanup(server.stop_server)
        transport = _mod_transport.get_transport_from_url(server.get_url())
        self.assertFalse(transport.abspath('/').endswith('/~/'))
        self.assertTrue(transport.abspath('/').endswith('/'))
        del transport