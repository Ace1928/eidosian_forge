from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_reset_quota(self):
    self.requests_mock.register_uri('DELETE', FAKE_LBAAS_URL + 'quotas/' + FAKE_PRJ, status_code=200)
    ret = self.api.quota_reset(FAKE_PRJ)
    self.assertEqual(200, ret.status_code)