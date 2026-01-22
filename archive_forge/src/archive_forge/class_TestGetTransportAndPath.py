import base64
import os
import shutil
import sys
import tempfile
import warnings
from io import BytesIO
from typing import Dict
from unittest.mock import patch
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
import dulwich
from dulwich import client
from dulwich.tests import TestCase, skipIf
from ..client import (
from ..config import ConfigDict
from ..objects import Commit, Tree
from ..pack import pack_objects_to_data, write_pack_data, write_pack_objects
from ..protocol import TCP_GIT_PORT, Protocol
from ..repo import MemoryRepo, Repo
from .utils import open_repo, setup_warning_catcher, tear_down_repo
class TestGetTransportAndPath(TestCase):

    def test_tcp(self):
        c, path = get_transport_and_path('git://foo.com/bar/baz')
        self.assertIsInstance(c, TCPGitClient)
        self.assertEqual('foo.com', c._host)
        self.assertEqual(TCP_GIT_PORT, c._port)
        self.assertEqual('/bar/baz', path)

    def test_tcp_port(self):
        c, path = get_transport_and_path('git://foo.com:1234/bar/baz')
        self.assertIsInstance(c, TCPGitClient)
        self.assertEqual('foo.com', c._host)
        self.assertEqual(1234, c._port)
        self.assertEqual('/bar/baz', path)

    def test_git_ssh_explicit(self):
        c, path = get_transport_and_path('git+ssh://foo.com/bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo.com', c.host)
        self.assertEqual(None, c.port)
        self.assertEqual(None, c.username)
        self.assertEqual('/bar/baz', path)

    def test_ssh_explicit(self):
        c, path = get_transport_and_path('ssh://foo.com/bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo.com', c.host)
        self.assertEqual(None, c.port)
        self.assertEqual(None, c.username)
        self.assertEqual('/bar/baz', path)

    def test_ssh_port_explicit(self):
        c, path = get_transport_and_path('git+ssh://foo.com:1234/bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo.com', c.host)
        self.assertEqual(1234, c.port)
        self.assertEqual('/bar/baz', path)

    def test_username_and_port_explicit_unknown_scheme(self):
        c, path = get_transport_and_path('unknown://git@server:7999/dply/stuff.git')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('unknown', c.host)
        self.assertEqual('//git@server:7999/dply/stuff.git', path)

    def test_username_and_port_explicit(self):
        c, path = get_transport_and_path('ssh://git@server:7999/dply/stuff.git')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('git', c.username)
        self.assertEqual('server', c.host)
        self.assertEqual(7999, c.port)
        self.assertEqual('/dply/stuff.git', path)

    def test_ssh_abspath_doubleslash(self):
        c, path = get_transport_and_path('git+ssh://foo.com//bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo.com', c.host)
        self.assertEqual(None, c.port)
        self.assertEqual(None, c.username)
        self.assertEqual('//bar/baz', path)

    def test_ssh_port(self):
        c, path = get_transport_and_path('git+ssh://foo.com:1234/bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo.com', c.host)
        self.assertEqual(1234, c.port)
        self.assertEqual('/bar/baz', path)

    def test_ssh_implicit(self):
        c, path = get_transport_and_path('foo:/bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo', c.host)
        self.assertEqual(None, c.port)
        self.assertEqual(None, c.username)
        self.assertEqual('/bar/baz', path)

    def test_ssh_host(self):
        c, path = get_transport_and_path('foo.com:/bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo.com', c.host)
        self.assertEqual(None, c.port)
        self.assertEqual(None, c.username)
        self.assertEqual('/bar/baz', path)

    def test_ssh_user_host(self):
        c, path = get_transport_and_path('user@foo.com:/bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo.com', c.host)
        self.assertEqual(None, c.port)
        self.assertEqual('user', c.username)
        self.assertEqual('/bar/baz', path)

    def test_ssh_relpath(self):
        c, path = get_transport_and_path('foo:bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo', c.host)
        self.assertEqual(None, c.port)
        self.assertEqual(None, c.username)
        self.assertEqual('bar/baz', path)

    def test_ssh_host_relpath(self):
        c, path = get_transport_and_path('foo.com:bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo.com', c.host)
        self.assertEqual(None, c.port)
        self.assertEqual(None, c.username)
        self.assertEqual('bar/baz', path)

    def test_ssh_user_host_relpath(self):
        c, path = get_transport_and_path('user@foo.com:bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo.com', c.host)
        self.assertEqual(None, c.port)
        self.assertEqual('user', c.username)
        self.assertEqual('bar/baz', path)

    def test_local(self):
        c, path = get_transport_and_path('foo.bar/baz')
        self.assertIsInstance(c, LocalGitClient)
        self.assertEqual('foo.bar/baz', path)

    @skipIf(sys.platform != 'win32', 'Behaviour only happens on windows.')
    def test_local_abs_windows_path(self):
        c, path = get_transport_and_path('C:\\foo.bar\\baz')
        self.assertIsInstance(c, LocalGitClient)
        self.assertEqual('C:\\foo.bar\\baz', path)

    def test_error(self):
        c, path = get_transport_and_path('prospero://bar/baz')
        self.assertIsInstance(c, SSHGitClient)

    def test_http(self):
        url = 'https://github.com/jelmer/dulwich'
        c, path = get_transport_and_path(url)
        self.assertIsInstance(c, HttpGitClient)
        self.assertEqual('/jelmer/dulwich', path)

    def test_http_auth(self):
        url = 'https://user:passwd@github.com/jelmer/dulwich'
        c, path = get_transport_and_path(url)
        self.assertIsInstance(c, HttpGitClient)
        self.assertEqual('/jelmer/dulwich', path)
        self.assertEqual('user', c._username)
        self.assertEqual('passwd', c._password)

    def test_http_auth_with_username(self):
        url = 'https://github.com/jelmer/dulwich'
        c, path = get_transport_and_path(url, username='user2', password='blah')
        self.assertIsInstance(c, HttpGitClient)
        self.assertEqual('/jelmer/dulwich', path)
        self.assertEqual('user2', c._username)
        self.assertEqual('blah', c._password)

    def test_http_auth_with_username_and_in_url(self):
        url = 'https://user:passwd@github.com/jelmer/dulwich'
        c, path = get_transport_and_path(url, username='user2', password='blah')
        self.assertIsInstance(c, HttpGitClient)
        self.assertEqual('/jelmer/dulwich', path)
        self.assertEqual('user', c._username)
        self.assertEqual('passwd', c._password)

    def test_http_no_auth(self):
        url = 'https://github.com/jelmer/dulwich'
        c, path = get_transport_and_path(url)
        self.assertIsInstance(c, HttpGitClient)
        self.assertEqual('/jelmer/dulwich', path)
        self.assertIs(None, c._username)
        self.assertIs(None, c._password)