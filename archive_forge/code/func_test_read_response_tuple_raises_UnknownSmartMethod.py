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
def test_read_response_tuple_raises_UnknownSmartMethod(self):
    """read_response_tuple raises UnknownSmartMethod if the server replied
        with 'UnknownMethod'.
        """
    headers = b'\x00\x00\x00\x02de'
    response_status = b'oE'
    args = b's\x00\x00\x00 l13:UnknownMethod11:method-namee'
    end = b'e'
    message_bytes = headers + response_status + args + end
    decoder, response_handler = self.make_conventional_response_decoder()
    decoder.accept_bytes(message_bytes)
    error = self.assertRaises(errors.UnknownSmartMethod, response_handler.read_response_tuple)
    self.assertEqual(b'method-name', error.verb)