import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
class TestSmartRequestHandlerErrorTranslation(TestCase):
    """Tests that SmartServerRequestHandler will translate exceptions raised by
    a SmartServerRequest into FailedSmartServerResponses.
    """

    def assertNoResponse(self, handler):
        self.assertEqual(None, handler.response)

    def assertResponseIsTranslatedError(self, handler):
        expected_translation = (b'NoSuchFile', b'xyzzy')
        self.assertEqual(request.FailedSmartServerResponse(expected_translation), handler.response)

    def test_error_translation_from_args_received(self):
        handler = request.SmartServerRequestHandler(None, {b'foo': DoErrorRequest}, '/')
        handler.args_received((b'foo',))
        self.assertResponseIsTranslatedError(handler)

    def test_error_translation_from_chunk_received(self):
        handler = request.SmartServerRequestHandler(None, {b'foo': ChunkErrorRequest}, '/')
        handler.args_received((b'foo',))
        self.assertNoResponse(handler)
        handler.accept_body(b'bytes')
        self.assertResponseIsTranslatedError(handler)

    def test_error_translation_from_end_received(self):
        handler = request.SmartServerRequestHandler(None, {b'foo': EndErrorRequest}, '/')
        handler.args_received((b'foo',))
        self.assertNoResponse(handler)
        handler.end_received()
        self.assertResponseIsTranslatedError(handler)

    def test_unexpected_error_translation(self):
        handler = request.SmartServerRequestHandler(None, {b'foo': DoUnexpectedErrorRequest}, '/')
        handler.args_received((b'foo',))
        self.assertEqual(request.FailedSmartServerResponse((b'error', b'KeyError', b'1')), handler.response)