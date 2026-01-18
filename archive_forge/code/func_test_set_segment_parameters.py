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
def test_set_segment_parameters(self):
    """Segment parameters can be set and show up in base."""
    transport = self.get_transport('foo')
    orig_base = transport.base
    transport.set_segment_parameter('arm', 'board')
    self.assertEqual('%s,arm=board' % orig_base, transport.base)
    self.assertEqual({'arm': 'board'}, transport.get_segment_parameters())
    transport.set_segment_parameter('arm', None)
    transport.set_segment_parameter('nonexistant', None)
    self.assertEqual({}, transport.get_segment_parameters())
    self.assertEqual(orig_base, transport.base)