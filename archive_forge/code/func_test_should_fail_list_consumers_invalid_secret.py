from barbicanclient import client
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_delete_secret_for_responses
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.tests.v1.test_secrets import SecretData
from barbicanclient.v1 import secrets
from oslo_serialization import jsonutils
def test_should_fail_list_consumers_invalid_secret(self):
    self.assertRaises(ValueError, self.manager.list_consumers, **{'secret_ref': '12345'})