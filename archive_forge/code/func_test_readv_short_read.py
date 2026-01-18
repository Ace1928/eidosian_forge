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
def test_readv_short_read(self):
    transport = self.get_transport()
    if transport.is_readonly():
        with open('a', 'w') as f:
            f.write('0123456789')
    else:
        transport.put_bytes('a', b'01234567890')
    self.assertListRaises((errors.ShortReadvError, errors.InvalidRange, AssertionError), transport.readv, 'a', [(1, 1), (8, 10)])
    self.assertListRaises((errors.ShortReadvError, errors.InvalidRange), transport.readv, 'a', [(12, 2)])