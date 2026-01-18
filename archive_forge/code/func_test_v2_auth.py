import copy
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
import testtools
import barbicanclient.barbican
def test_v2_auth(self):
    self.responses.get(V2_URL, body=V2_VERSION_ENTRY)
    self.responses.post('{0}/tokens'.format(V2_URL), json=generate_v2_project_scoped_token())
    self._delete_secret('v2')