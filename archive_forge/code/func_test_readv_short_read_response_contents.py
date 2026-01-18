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
def test_readv_short_read_response_contents(self):
    """'readv' when a short read occurs sets the response appropriately."""
    self.build_tree(['a-file'])
    handler = self.build_handler(self.get_readonly_transport())
    handler.args_received((b'readv', b'a-file'))
    handler.accept_body(b'100,1')
    handler.end_of_body()
    self.assertTrue(handler.finished_reading)
    self.assertEqual((b'ShortReadvError', b'./a-file', b'100', b'1', b'0'), handler.response.args)
    self.assertEqual(None, handler.response.body)