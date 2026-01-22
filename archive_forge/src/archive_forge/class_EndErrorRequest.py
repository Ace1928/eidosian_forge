import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
class EndErrorRequest(request.SmartServerRequest):
    """A request that raises an error from self.do_end()."""

    def do(self):
        """No-op."""
        pass

    def do_chunk(self, bytes):
        """No-op."""
        pass

    def do_end(self):
        raise transport.NoSuchFile('xyzzy')