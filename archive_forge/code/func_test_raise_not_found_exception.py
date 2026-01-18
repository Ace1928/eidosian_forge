import json
from unittest import mock
import uuid
from openstack import exceptions
from openstack.tests.unit import base
def test_raise_not_found_exception(self):
    response = mock.Mock()
    response.status_code = 404
    response.headers = {'content-type': 'application/json', 'x-openstack-request-id': uuid.uuid4().hex}
    exc = self.assertRaises(exceptions.NotFoundException, self._do_raise, response, error_message=self.message)
    self.assertEqual(self.message, exc.message)
    self.assertEqual(response.status_code, exc.status_code)
    self.assertEqual(response.headers.get('x-openstack-request-id'), exc.request_id)