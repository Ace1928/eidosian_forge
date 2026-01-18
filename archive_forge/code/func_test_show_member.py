from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_show_member(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'pools/' + FAKE_PO + '/members/' + FAKE_ME, json=SINGLE_ME_RESP, status_code=200)
    ret = self.api.member_show(pool_id=FAKE_PO, member_id=FAKE_ME)
    self.assertEqual(SINGLE_ME_RESP['member'], ret)