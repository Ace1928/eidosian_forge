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
def test_rmdir_empty_but_similar_prefix(self):
    """rmdir does not get confused by sibling paths.

        A naive implementation of MemoryTransport would refuse to rmdir
        ".bzr/branch" if there is a ".bzr/branch-format" directory, because it
        uses "path.startswith(dir)" on all file paths to determine if directory
        is empty.
        """
    t = self.get_transport()
    if t.is_readonly():
        return
    t.mkdir('foo')
    t.put_bytes('foo-bar', b'')
    t.mkdir('foo-baz')
    t.rmdir('foo')
    self.assertRaises((NoSuchFile, PathError), t.rmdir, 'foo')
    self.assertTrue(t.has('foo-bar'))