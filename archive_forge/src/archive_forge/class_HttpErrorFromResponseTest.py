import unittest
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
class HttpErrorFromResponseTest(unittest.TestCase):
    """Tests for exceptions.HttpError.FromResponse."""

    def testBadRequest(self):
        err = exceptions.HttpError.FromResponse(_MakeResponse(400))
        self.assertIsInstance(err, exceptions.HttpError)
        self.assertIsInstance(err, exceptions.HttpBadRequestError)
        self.assertEquals(err.status_code, 400)

    def testUnauthorized(self):
        err = exceptions.HttpError.FromResponse(_MakeResponse(401))
        self.assertIsInstance(err, exceptions.HttpError)
        self.assertIsInstance(err, exceptions.HttpUnauthorizedError)
        self.assertEquals(err.status_code, 401)

    def testForbidden(self):
        err = exceptions.HttpError.FromResponse(_MakeResponse(403))
        self.assertIsInstance(err, exceptions.HttpError)
        self.assertIsInstance(err, exceptions.HttpForbiddenError)
        self.assertEquals(err.status_code, 403)

    def testExceptionMessageIncludesErrorDetails(self):
        err = exceptions.HttpError.FromResponse(_MakeResponse(403))
        self.assertIn('403', repr(err))
        self.assertIn('http://www.google.com', repr(err))
        self.assertIn('{"field": "abc"}', repr(err))

    def testNotFound(self):
        err = exceptions.HttpError.FromResponse(_MakeResponse(404))
        self.assertIsInstance(err, exceptions.HttpError)
        self.assertIsInstance(err, exceptions.HttpNotFoundError)
        self.assertEquals(err.status_code, 404)

    def testConflict(self):
        err = exceptions.HttpError.FromResponse(_MakeResponse(409))
        self.assertIsInstance(err, exceptions.HttpError)
        self.assertIsInstance(err, exceptions.HttpConflictError)
        self.assertEquals(err.status_code, 409)

    def testUnknownStatus(self):
        err = exceptions.HttpError.FromResponse(_MakeResponse(499))
        self.assertIsInstance(err, exceptions.HttpError)
        self.assertEquals(err.status_code, 499)

    def testMalformedStatus(self):
        err = exceptions.HttpError.FromResponse(_MakeResponse('BAD'))
        self.assertIsInstance(err, exceptions.HttpError)