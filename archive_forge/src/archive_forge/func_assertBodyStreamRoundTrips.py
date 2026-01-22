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
def assertBodyStreamRoundTrips(self, body_stream):
    """Assert that body_stream is the same after being serialised and
        deserialised.
        """
    out_stream = BytesIO()
    protocol._send_stream(body_stream, out_stream.write)
    decoder = protocol.ChunkedBodyDecoder()
    decoder.accept_bytes(out_stream.getvalue())
    decoded_stream = list(iter(decoder.read_next_chunk, None))
    self.assertEqual(body_stream, decoded_stream)