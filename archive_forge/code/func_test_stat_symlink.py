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
def test_stat_symlink(self):
    t = self.get_transport()
    try:
        t.symlink('target', 'link')
    except TransportNotPossible:
        raise TestSkipped('symlinks not supported')
    t2 = t.clone('link')
    st = t2.stat('')
    self.assertTrue(stat.S_ISLNK(st.st_mode))