import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
def test_should_store_via_constructor(self):
    data = {'secret_ref': self.entity_href}
    self.responses.post(self.entity_base + '/', json=data)
    secret = self.manager.create(name=self.secret.name, payload=self.secret.payload)
    secret_href = secret.store()
    self.assertEqual(self.entity_href, secret_href)
    secret_req = jsonutils.loads(self.responses.last_request.text)
    self.assertEqual(self.secret.name, secret_req['name'])
    self.assertEqual(self.secret.payload, secret_req['payload'])