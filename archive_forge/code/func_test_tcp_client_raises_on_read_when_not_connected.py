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
def test_tcp_client_raises_on_read_when_not_connected(self):
    client_medium = medium.SmartTCPClientMedium(None, None, None)
    self.assertRaises(errors.MediumNotConnected, client_medium.read_bytes, 0)
    self.assertRaises(errors.MediumNotConnected, client_medium.read_bytes, 1)