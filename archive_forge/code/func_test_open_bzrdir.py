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
def test_open_bzrdir(self):
    """Open an existing bzrdir over smart transport"""
    transport = self.transport
    t = self.backing_transport
    bzrdir.BzrDirFormat.get_default_format().initialize_on_transport(t)
    result_dir = controldir.ControlDir.open_containing_from_transport(transport)