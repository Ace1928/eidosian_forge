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
def test_rename_dir_succeeds(self):
    t = self.get_transport()
    if t.is_readonly():
        self.assertRaises((TransportNotPossible, NotImplementedError), t.rename, 'foo', 'bar')
        return
    t.mkdir('adir')
    t.mkdir('adir/asubdir')
    t.rename('adir', 'bdir')
    self.assertTrue(t.has('bdir/asubdir'))
    self.assertFalse(t.has('adir'))