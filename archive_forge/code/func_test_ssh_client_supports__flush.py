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
def test_ssh_client_supports__flush(self):
    from io import BytesIO
    input = BytesIO()
    output = BytesIO()
    flush_calls = []

    def logging_flush():
        flush_calls.append('flush')
    output.flush = logging_flush
    vendor = BytesIOSSHVendor(input, output)
    client_medium = medium.SmartSSHClientMedium('base', medium.SSHParams('a hostname'), vendor=vendor)
    client_medium._accept_bytes(b'abc')
    client_medium._flush()
    client_medium.disconnect()
    self.assertEqual(['flush'], flush_calls)