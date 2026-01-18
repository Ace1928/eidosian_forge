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
def test_segment_parameters(self):
    """Segment parameters should be stripped and stored in
        transport.get_segment_parameters()."""
    base_url = self._server.get_url()
    parameters = {'key1': 'val1', 'key2': 'val2'}
    url = urlutils.join_segment_parameters(base_url, parameters)
    transport = _mod_transport.get_transport_from_url(url)
    self.assertEqual(parameters, transport.get_segment_parameters())