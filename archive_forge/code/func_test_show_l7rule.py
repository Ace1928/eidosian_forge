from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_show_l7rule(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'l7policies/' + FAKE_L7PO + '/rules/' + FAKE_L7RU, json=SINGLE_L7RU_RESP, status_code=200)
    ret = self.api.l7rule_show(FAKE_L7RU, FAKE_L7PO)
    self.assertEqual(SINGLE_L7RU_RESP['rule'], ret)