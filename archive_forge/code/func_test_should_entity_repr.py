import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
def test_should_entity_repr(self):
    secret_obj = self.manager.create(name=self.secret.name)
    self.assertIn('name="{0}"'.format(self.secret.name), repr(secret_obj))