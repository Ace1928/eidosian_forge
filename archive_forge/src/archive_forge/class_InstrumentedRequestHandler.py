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
class InstrumentedRequestHandler:
    """Test Double of SmartServerRequestHandler."""

    def __init__(self):
        self.calls = []
        self.finished_reading = False

    def no_body_received(self):
        self.calls.append(('no_body_received',))

    def end_received(self):
        self.calls.append(('end_received',))
        self.finished_reading = True

    def args_received(self, args):
        self.calls.append(('args_received', args))

    def accept_body(self, bytes):
        self.calls.append(('accept_body', bytes))

    def end_of_body(self):
        self.calls.append(('end_of_body',))
        self.finished_reading = True

    def post_body_error_received(self, error_args):
        self.calls.append(('post_body_error_received', error_args))