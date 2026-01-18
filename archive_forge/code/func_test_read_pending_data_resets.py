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
def test_read_pending_data_resets(self):
    """read_pending_data does not return the same bytes twice."""
    decoder = protocol.ChunkedBodyDecoder()
    decoder.accept_bytes(b'chunked\n')
    chunk_one = b'3\naaa'
    chunk_two = b'3\nbbb'
    finish = b'END\n'
    decoder.accept_bytes(chunk_one)
    self.assertEqual(b'aaa', decoder.read_next_chunk())
    decoder.accept_bytes(chunk_two)
    self.assertEqual(b'bbb', decoder.read_next_chunk())
    self.assertEqual(None, decoder.read_next_chunk())