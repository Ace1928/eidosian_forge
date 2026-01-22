import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
class TestFormatException(unittest.TestCase):

    def _test_format_exception(self, exception, debug=False):
        fake_exc_info = (None, exception, None)
        return wsme_api.format_exception(fake_exc_info, debug=debug)

    def test_format_client_exception(self):
        faultstring = 'boom'
        ret = self._test_format_exception(exc.ClientSideError(faultstring))
        self.assertIsNone(ret['debuginfo'])
        self.assertEqual('Client', ret['faultcode'])
        self.assertEqual(faultstring, ret['faultstring'])

    def test_format_client_exception_unicode(self):
        faultstring = u'Ã£o'
        ret = self._test_format_exception(exc.ClientSideError(faultstring))
        self.assertIsNone(ret['debuginfo'])
        self.assertEqual('Client', ret['faultcode'])
        self.assertEqual(faultstring, ret['faultstring'])

    def test_format_client_exception_with_faultcode(self):
        faultcode = 'AccessDenied'
        faultstring = 'boom'
        ret = self._test_format_exception(exc.ClientSideError(faultstring, faultcode=faultcode))
        self.assertIsNone(ret['debuginfo'])
        self.assertEqual('AccessDenied', ret['faultcode'])
        self.assertEqual(faultstring, ret['faultstring'])

    def test_format_server_exception(self):
        faultstring = 'boom'
        ret = self._test_format_exception(Exception(faultstring))
        self.assertIsNone(ret['debuginfo'])
        self.assertEqual('Server', ret['faultcode'])
        self.assertEqual(faultstring, ret['faultstring'])

    def test_format_server_exception_unicode(self):
        faultstring = u'Ã£o'
        ret = self._test_format_exception(Exception(faultstring))
        self.assertIsNone(ret['debuginfo'])
        self.assertEqual('Server', ret['faultcode'])
        self.assertEqual(faultstring, ret['faultstring'])

    def test_format_server_exception_with_faultcode(self):
        faultstring = 'boom'
        exception = Exception(faultstring)
        exception.faultcode = 'ServerError'
        ret = self._test_format_exception(exception)
        self.assertIsNone(ret['debuginfo'])
        self.assertEqual('ServerError', ret['faultcode'])
        self.assertEqual(faultstring, ret['faultstring'])

    def test_format_server_exception_debug(self):
        faultstring = 'boom'
        ret = self._test_format_exception(Exception(faultstring), debug=True)
        self.assertIsNotNone(ret['debuginfo'])
        self.assertEqual('Server', ret['faultcode'])
        self.assertEqual(faultstring, ret['faultstring'])