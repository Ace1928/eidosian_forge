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
def test_ssh_client_connects_on_first_use(self):
    output = BytesIO()
    vendor = BytesIOSSHVendor(BytesIO(), output)
    ssh_params = medium.SSHParams('a hostname', 'a port', 'a username', 'a password', 'bzr')
    client_medium = medium.SmartSSHClientMedium('base', ssh_params, vendor)
    client_medium._accept_bytes(b'abc')
    self.assertEqual(b'abc', output.getvalue())
    self.assertEqual([('connect_ssh', 'a username', 'a password', 'a hostname', 'a port', ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes'])], vendor.calls)