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
class TestServerHooks(SmartTCPTests):

    def capture_server_call(self, backing_urls, public_url):
        """Record a server_started|stopped hook firing."""
        self.hook_calls.append((backing_urls, public_url))

    def test_server_started_hook_memory(self):
        """The server_started hook fires when the server is started."""
        self.hook_calls = []
        _mod_server.SmartTCPServer.hooks.install_named_hook('server_started', self.capture_server_call, None)
        self.start_server()
        self.transport.has('.')
        self.assertEqual([([self.backing_transport.base], self.transport.base)], self.hook_calls)

    def test_server_started_hook_file(self):
        """The server_started hook fires when the server is started."""
        self.hook_calls = []
        _mod_server.SmartTCPServer.hooks.install_named_hook('server_started', self.capture_server_call, None)
        self.start_server(backing_transport=_mod_transport.get_transport_from_path('.'))
        self.transport.has('.')
        self.assertEqual([([self.backing_transport.base, self.backing_transport.external_url()], self.transport.base)], self.hook_calls)

    def test_server_stopped_hook_simple_memory(self):
        """The server_stopped hook fires when the server is stopped."""
        self.hook_calls = []
        _mod_server.SmartTCPServer.hooks.install_named_hook('server_stopped', self.capture_server_call, None)
        self.start_server()
        result = [([self.backing_transport.base], self.transport.base)]
        self.assertEqual([], self.hook_calls)
        self.transport.has('.')
        self.assertEqual([], self.hook_calls)
        self.stop_server()
        self.assertEqual(result, self.hook_calls)

    def test_server_stopped_hook_simple_file(self):
        """The server_stopped hook fires when the server is stopped."""
        self.hook_calls = []
        _mod_server.SmartTCPServer.hooks.install_named_hook('server_stopped', self.capture_server_call, None)
        self.start_server(backing_transport=_mod_transport.get_transport_from_path('.'))
        result = [([self.backing_transport.base, self.backing_transport.external_url()], self.transport.base)]
        self.assertEqual([], self.hook_calls)
        self.transport.has('.')
        self.assertEqual([], self.hook_calls)
        self.stop_server()
        self.assertEqual(result, self.hook_calls)