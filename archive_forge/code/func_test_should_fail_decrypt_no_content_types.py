import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
def test_should_fail_decrypt_no_content_types(self):
    data = self.secret.get_dict(self.entity_href)
    self.responses.get(self.entity_href, json=data)
    secret = self.manager.get(secret_ref=self.entity_href)
    self.assertIsNone(secret.payload)