import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
class TestConnectedTransport(tests.TestCase):
    """Tests for connected to remote server transports"""

    def test_parse_url(self):
        t = transport.ConnectedTransport('http://simple.example.com/home/source')
        self.assertEqual(t._parsed_url.host, 'simple.example.com')
        self.assertEqual(t._parsed_url.port, None)
        self.assertEqual(t._parsed_url.path, '/home/source/')
        self.assertTrue(t._parsed_url.user is None)
        self.assertTrue(t._parsed_url.password is None)
        self.assertEqual(t.base, 'http://simple.example.com/home/source/')

    def test_parse_url_with_at_in_user(self):
        t = transport.ConnectedTransport('ftp://user@host.com@www.host.com/')
        self.assertEqual(t._parsed_url.user, 'user@host.com')

    def test_parse_quoted_url(self):
        t = transport.ConnectedTransport('http://ro%62ey:h%40t@ex%41mple.com:2222/path')
        self.assertEqual(t._parsed_url.host, 'exAmple.com')
        self.assertEqual(t._parsed_url.port, 2222)
        self.assertEqual(t._parsed_url.user, 'robey')
        self.assertEqual(t._parsed_url.password, 'h@t')
        self.assertEqual(t._parsed_url.path, '/path/')
        self.assertEqual(t.base, 'http://ro%62ey@ex%41mple.com:2222/path/')

    def test_parse_invalid_url(self):
        self.assertRaises(urlutils.InvalidURL, transport.ConnectedTransport, 'sftp://lily.org:~janneke/public/bzr/gub')

    def test_relpath(self):
        t = transport.ConnectedTransport('sftp://user@host.com/abs/path')
        self.assertEqual(t.relpath('sftp://user@host.com/abs/path/sub'), 'sub')
        self.assertRaises(errors.PathNotChild, t.relpath, 'http://user@host.com/abs/path/sub')
        self.assertRaises(errors.PathNotChild, t.relpath, 'sftp://user2@host.com/abs/path/sub')
        self.assertRaises(errors.PathNotChild, t.relpath, 'sftp://user@otherhost.com/abs/path/sub')
        self.assertRaises(errors.PathNotChild, t.relpath, 'sftp://user@host.com:33/abs/path/sub')
        t = transport.ConnectedTransport('sftp://host.com/abs/path')
        self.assertEqual(t.relpath('sftp://host.com/abs/path/sub'), 'sub')
        t = transport.ConnectedTransport('sftp://host.com/dev/%path')
        self.assertEqual(t.relpath('sftp://host.com/dev/%path/sub'), 'sub')

    def test_connection_sharing_propagate_credentials(self):
        t = transport.ConnectedTransport('ftp://user@host.com/abs/path')
        self.assertEqual('user', t._parsed_url.user)
        self.assertEqual('host.com', t._parsed_url.host)
        self.assertIs(None, t._get_connection())
        self.assertIs(None, t._parsed_url.password)
        c = t.clone('subdir')
        self.assertIs(None, c._get_connection())
        self.assertIs(None, t._parsed_url.password)
        password = 'secret'
        connection = object()
        t._set_connection(connection, password)
        self.assertIs(connection, t._get_connection())
        self.assertIs(password, t._get_credentials())
        self.assertIs(connection, c._get_connection())
        self.assertIs(password, c._get_credentials())
        new_password = 'even more secret'
        c._update_credentials(new_password)
        self.assertIs(connection, t._get_connection())
        self.assertIs(new_password, t._get_credentials())
        self.assertIs(connection, c._get_connection())
        self.assertIs(new_password, c._get_credentials())