import json
from unittest import mock
import uuid
from openstack import exceptions
from openstack.tests.unit import base
class Test_HttpException(base.TestCase):

    def setUp(self):
        super(Test_HttpException, self).setUp()
        self.message = 'mayday'

    def _do_raise(self, *args, **kwargs):
        raise exceptions.HttpException(*args, **kwargs)

    def test_message(self):
        exc = self.assertRaises(exceptions.HttpException, self._do_raise, self.message)
        self.assertEqual(self.message, exc.message)

    def test_details(self):
        details = 'some details'
        exc = self.assertRaises(exceptions.HttpException, self._do_raise, self.message, details=details)
        self.assertEqual(self.message, exc.message)
        self.assertEqual(details, exc.details)

    def test_http_status(self):
        http_status = 123
        exc = self.assertRaises(exceptions.HttpException, self._do_raise, self.message, http_status=http_status)
        self.assertEqual(self.message, exc.message)
        self.assertEqual(http_status, exc.status_code)