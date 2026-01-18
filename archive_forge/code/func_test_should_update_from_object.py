import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
def test_should_update_from_object(self, secref_ref=None):
    secref_ref = secref_ref or self.entity_href
    data = {'secret_ref': secref_ref}
    self.responses.post(self.entity_base + '/', json=data)
    secret = self.manager.create()
    secret.payload = None
    secret.store()
    self.assertEqual(secref_ref, secret.secret_ref)
    text_payload = 'time for an ice cold 🍺'
    self.responses.put(self.entity_href, status_code=204)
    secret.payload = text_payload
    secret.update()
    self.assertEqual(self.entity_href, self.responses.last_request.url)
    self.assertEqual(text_payload, secret.payload)