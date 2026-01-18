import json
from unittest import mock
import uuid
from openstack import exceptions
from openstack.tests.unit import base
def test_raise_no_exception(self):
    response = mock.Mock()
    response.status_code = 200
    self.assertIsNone(self._do_raise(response))