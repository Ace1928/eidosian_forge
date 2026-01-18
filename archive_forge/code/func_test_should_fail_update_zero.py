import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
def test_should_fail_update_zero(self):
    data = {'secret_ref': self.entity_href}
    self.responses.post(self.entity_base + '/', json=data)
    secret = self.manager.create()
    secret.payload = None
    secret.store()
    self.responses.put(self.entity_href, status_code=204)
    secret.payload = 0
    self.assertRaises(exceptions.PayloadException, secret.update)