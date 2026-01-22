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
def assertOffsetSerialisation(self, expected_offsets, expected_serialised, requester):
    """Check that smart (de)serialises offsets as expected.

        We check both serialisation and deserialisation at the same time
        to ensure that the round tripping cannot skew: both directions should
        be as expected.

        :param expected_offsets: a readv offset list.
        :param expected_seralised: an expected serial form of the offsets.
        """
    readv_cmd = vfs.ReadvRequest(None, '/')
    offsets = readv_cmd._deserialise_offsets(expected_serialised)
    self.assertEqual(expected_offsets, offsets)
    serialised = requester._serialise_offsets(offsets)
    self.assertEqual(expected_serialised, serialised)