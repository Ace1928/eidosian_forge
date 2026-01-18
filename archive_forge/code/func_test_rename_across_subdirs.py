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
def test_rename_across_subdirs(self):
    t = self.get_transport()
    if t.is_readonly():
        raise TestNotApplicable('transport is readonly')
    t.mkdir('a')
    t.mkdir('b')
    ta = t.clone('a')
    tb = t.clone('b')
    ta.put_bytes('f', b'aoeu')
    ta.rename('f', '../b/f')
    self.assertTrue(tb.has('f'))
    self.assertFalse(ta.has('f'))
    self.assertTrue(t.has('b/f'))