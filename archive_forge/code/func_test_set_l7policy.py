from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_set_l7policy(self):
    self.requests_mock.register_uri('PUT', FAKE_LBAAS_URL + 'l7policies/' + FAKE_L7PO, json=SINGLE_L7PO_UPDATE, status_code=200)
    ret = self.api.l7policy_set(FAKE_L7PO, json=SINGLE_L7PO_UPDATE)
    self.assertEqual(SINGLE_L7PO_UPDATE, ret)