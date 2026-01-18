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
def test_ssh_client_disconnect_allows_reconnection(self):
    input = BytesIO()
    output = BytesIO()
    vendor = BytesIOSSHVendor(input, output)
    client_medium = medium.SmartSSHClientMedium('base', medium.SSHParams('a hostname'), vendor)
    client_medium._accept_bytes(b'abc')
    client_medium.disconnect()
    input2 = BytesIO()
    output2 = BytesIO()
    vendor.read_from = input2
    vendor.write_to = output2
    client_medium._accept_bytes(b'abc')
    client_medium.disconnect()
    self.assertTrue(input.closed)
    self.assertTrue(output.closed)
    self.assertTrue(input2.closed)
    self.assertTrue(output2.closed)
    self.assertEqual([('connect_ssh', None, None, 'a hostname', None, ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes']), ('close',), ('connect_ssh', None, None, 'a hostname', None, ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes']), ('close',)], vendor.calls)