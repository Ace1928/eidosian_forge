import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
def test_should_decrypt_using_stripped_uuid(self):
    bad_href = 'http://badsite.com/' + self.entity_id
    self.test_should_decrypt(bad_href)