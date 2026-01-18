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
def test_read_response_tuple_error(self):
    """If the response has an error, it is raised as an exception."""
    headers = b'\x00\x00\x00\x02de'
    response_status = b'oE'
    args = b's\x00\x00\x00\x1al9:first arg10:second arge'
    end = b'e'
    message_bytes = headers + response_status + args + end
    decoder, response_handler = self.make_conventional_response_decoder()
    decoder.accept_bytes(message_bytes)
    error = self.assertRaises(errors.ErrorFromSmartServer, response_handler.read_response_tuple)
    self.assertEqual((b'first arg', b'second arg'), error.error_tuple)