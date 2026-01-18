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
def test_one_chunk(self):
    """A body in a single chunk is decoded correctly."""
    decoder = protocol.ChunkedBodyDecoder()
    decoder.accept_bytes(b'chunked\n')
    chunk_length = b'f\n'
    chunk_content = b'123456789abcdef'
    finish = b'END\n'
    decoder.accept_bytes(chunk_length + chunk_content + finish)
    self.assertTrue(decoder.finished_reading)
    self.assertEqual(chunk_content, decoder.read_next_chunk())
    self.assertEqual(b'', decoder.unused_data)