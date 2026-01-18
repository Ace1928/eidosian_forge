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
def test_rename_dir_nonempty(self):
    """Attempting to replace a nonemtpy directory should fail"""
    t = self.get_transport()
    if t.is_readonly():
        self.assertRaises((TransportNotPossible, NotImplementedError), t.rename, 'foo', 'bar')
        return
    t.mkdir('adir')
    t.mkdir('adir/asubdir')
    t.mkdir('bdir')
    t.mkdir('bdir/bsubdir')
    self.assertRaises(PathError, t.rename, 'bdir', 'adir')
    self.assertTrue(t.has('bdir/bsubdir'))
    self.assertFalse(t.has('adir/bdir'))
    self.assertFalse(t.has('adir/bsubdir'))