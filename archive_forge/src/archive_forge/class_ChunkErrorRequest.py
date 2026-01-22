import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
class ChunkErrorRequest(request.SmartServerRequest):
    """A request that raises an error from self.do_chunk()."""

    def do(self):
        """No-op."""
        pass

    def do_chunk(self, bytes):
        raise transport.NoSuchFile('xyzzy')