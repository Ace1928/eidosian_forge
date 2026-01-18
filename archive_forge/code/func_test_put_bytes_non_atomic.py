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
def test_put_bytes_non_atomic(self):
    """'put_...' should set finished_reading after reading the bytes."""
    handler = self.build_handler(self.get_transport())
    handler.args_received((b'put_non_atomic', b'a-file', b'', b'F', b''))
    self.assertFalse(handler.finished_reading)
    handler.accept_body(b'1234')
    self.assertFalse(handler.finished_reading)
    handler.accept_body(b'5678')
    handler.end_of_body()
    self.assertTrue(handler.finished_reading)
    self.assertEqual((b'ok',), handler.response.args)
    self.assertEqual(None, handler.response.body)