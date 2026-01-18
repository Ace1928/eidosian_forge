import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
def test_should_get_payload_only_when_content_type_is_set(self):
    """DEPRECATION WARNING

        Manually setting the payload_content_type is deprecated and will be
        removed in a future release.
        """
    m = self.responses.get(self.entity_href, request_headers={'Accept': 'application/json'}, json=self.secret.get_dict(self.entity_href))
    n = self.responses.get(self.entity_payload_href, request_headers={'Accept': 'text/plain'}, text=self.secret.payload)
    secret = self.manager.get(secret_ref=self.entity_href, payload_content_type=self.secret.payload_content_type)
    self.assertIsInstance(secret, secrets.Secret)
    self.assertEqual(self.entity_href, secret.secret_ref)
    self.assertFalse(m.called)
    self.assertFalse(n.called)
    self.assertEqual(self.secret.payload, secret.payload)
    self.assertFalse(m.called)
    self.assertTrue(n.called)
    self.assertEqual(self.entity_payload_href, n.last_request.url)