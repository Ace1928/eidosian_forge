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
def test_rmdir_not_empty(self):
    """Deleting a non-empty directory raises an exception

        sftp (and possibly others) don't give us a specific "directory not
        empty" exception -- we can just see that the operation failed.
        """
    t = self.get_transport()
    if t.is_readonly():
        return
    t.mkdir('adir')
    t.mkdir('adir/bdir')
    self.assertRaises(PathError, t.rmdir, 'adir')