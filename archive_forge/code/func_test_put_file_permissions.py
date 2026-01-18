import os
import stat
import sys
from io import BytesIO
from .. import errors, osutils, pyutils, tests
from .. import transport as _mod_transport
from .. import urlutils
from ..errors import ConnectionError, PathError, TransportNotPossible
from ..osutils import getcwd
from ..transport import (ConnectedTransport, FileExists, NoSuchFile, Transport,
from ..transport.memory import MemoryTransport
from ..transport.remote import RemoteTransport
from . import TestNotApplicable, TestSkipped, multiply_tests, test_server
from .test_transport import TestTransportImplementation
def test_put_file_permissions(self):
    t = self.get_transport()
    if t.is_readonly():
        return
    if not t._can_roundtrip_unix_modebits():
        return
    t.put_file('mode644', BytesIO(b'test text\n'), mode=420)
    self.assertTransportMode(t, 'mode644', 420)
    t.put_file('mode666', BytesIO(b'test text\n'), mode=438)
    self.assertTransportMode(t, 'mode666', 438)
    t.put_file('mode600', BytesIO(b'test text\n'), mode=384)
    self.assertTransportMode(t, 'mode600', 384)
    t.put_file('mode400', BytesIO(b'test text\n'), mode=256)
    self.assertTransportMode(t, 'mode400', 256)
    umask = osutils.get_umask()
    t.put_file('nomode', BytesIO(b'test text\n'), mode=None)
    self.assertTransportMode(t, 'nomode', 438 & ~umask)