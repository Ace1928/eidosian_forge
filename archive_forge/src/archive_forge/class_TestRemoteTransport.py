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
class TestRemoteTransport(tests.TestCase):

    def test_use_connection_factory(self):
        input = BytesIO(b'ok\n3\nbardone\n')
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        transport = remote.RemoteTransport('bzr://localhost/', medium=client_medium)
        client_medium._protocol_version = 1
        self.assertEqual(0, input.tell())
        self.assertEqual(b'', output.getvalue())
        self.assertEqual(b'bar', transport.get_bytes('foo'))
        self.assertEqual(13, input.tell())
        self.assertEqual(b'get\x01/foo\n', output.getvalue())

    def test__translate_error_readonly(self):
        """Sending a ReadOnlyError to _translate_error raises TransportNotPossible."""
        client_medium = medium.SmartSimplePipesClientMedium(None, None, 'base')
        transport = remote.RemoteTransport('bzr://localhost/', medium=client_medium)
        err = errors.ErrorFromSmartServer((b'ReadOnlyError',))
        self.assertRaises(errors.TransportNotPossible, transport._translate_error, err)