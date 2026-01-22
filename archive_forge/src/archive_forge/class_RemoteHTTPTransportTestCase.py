import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
class RemoteHTTPTransportTestCase(tests.TestCase):

    def test_remote_path_after_clone_child(self):
        base_transport = remote.RemoteHTTPTransport('bzr+http://host/path')
        new_transport = base_transport.clone('child_dir')
        self.assertEqual(base_transport._http_transport, new_transport._http_transport)
        self.assertEqual('child_dir/foo', new_transport._remote_path('foo'))
        self.assertEqual(b'child_dir/', new_transport._client.remote_path_from_transport(new_transport))

    def test_remote_path_unnormal_base(self):
        base_transport = remote.RemoteHTTPTransport('bzr+http://host/%7Ea/b')
        self.assertEqual('c', base_transport._remote_path('c'))

    def test_clone_unnormal_base(self):
        base_transport = remote.RemoteHTTPTransport('bzr+http://host/%7Ea/b')
        new_transport = base_transport.clone('c')
        self.assertEqual(base_transport.base + 'c/', new_transport.base)
        self.assertEqual(b'c/', new_transport._client.remote_path_from_transport(new_transport))

    def test__redirect_to(self):
        t = remote.RemoteHTTPTransport('bzr+http://www.example.com/foo')
        r = t._redirected_to('http://www.example.com/foo', 'http://www.example.com/bar')
        self.assertEqual(type(r), type(t))

    def test__redirect_sibling_protocol(self):
        t = remote.RemoteHTTPTransport('bzr+http://www.example.com/foo')
        r = t._redirected_to('http://www.example.com/foo', 'https://www.example.com/bar')
        self.assertEqual(type(r), type(t))
        self.assertStartsWith(r.base, 'bzr+https')

    def test__redirect_to_with_user(self):
        t = remote.RemoteHTTPTransport('bzr+http://joe@www.example.com/foo')
        r = t._redirected_to('http://www.example.com/foo', 'http://www.example.com/bar')
        self.assertEqual(type(r), type(t))
        self.assertEqual('joe', t._parsed_url.user)
        self.assertEqual(t._parsed_url.user, r._parsed_url.user)

    def test_redirected_to_same_host_different_protocol(self):
        t = remote.RemoteHTTPTransport('bzr+http://joe@www.example.com/foo')
        r = t._redirected_to('http://www.example.com/foo', 'bzr://www.example.com/foo')
        self.assertNotEqual(type(r), type(t))