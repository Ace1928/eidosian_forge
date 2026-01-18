import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
def test_post_connect(self):
    """Ensure the post_connect hook is called when _set_transport is"""
    calls = []
    transport.Transport.hooks.install_named_hook('post_connect', calls.append, None)
    t = self._get_connected_transport()
    self.assertLength(0, calls)
    t._set_connection('connection', 'auth')
    self.assertEqual(calls, [t])