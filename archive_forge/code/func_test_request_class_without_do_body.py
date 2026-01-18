import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
def test_request_class_without_do_body(self):
    """If a request has no body data, and the request's implementation does
        not override do_body, then no exception is raised.
        """
    handler = request.SmartServerRequestHandler(None, {b'foo': NoBodyRequest}, '/')
    handler.args_received((b'foo',))
    handler.end_received()