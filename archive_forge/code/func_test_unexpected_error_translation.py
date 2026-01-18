import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
def test_unexpected_error_translation(self):
    handler = request.SmartServerRequestHandler(None, {b'foo': DoUnexpectedErrorRequest}, '/')
    handler.args_received((b'foo',))
    self.assertEqual(request.FailedSmartServerResponse((b'error', b'KeyError', b'1')), handler.response)