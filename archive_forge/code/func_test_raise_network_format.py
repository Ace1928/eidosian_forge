import json
from unittest import mock
import uuid
from openstack import exceptions
from openstack.tests.unit import base
def test_raise_network_format(self):
    response = mock.Mock()
    response.status_code = 404
    response.headers = {'content-type': 'application/json'}
    response.json.return_value = {'NeutronError': {'message': self.message, 'type': 'FooNotFound', 'detail': ''}}
    exc = self.assertRaises(exceptions.NotFoundException, self._do_raise, response, error_message=self.message)
    self.assertEqual(response.status_code, exc.status_code)
    self.assertEqual(self.message, exc.details)
    self.assertIn(self.message, str(exc))