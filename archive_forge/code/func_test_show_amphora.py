from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_show_amphora(self):
    self.requests_mock.register_uri('GET', FAKE_OCTAVIA_URL + 'amphorae/' + FAKE_AMP, json=SINGLB_AMP_RESP, status_code=200)
    ret = self.api.amphora_show(FAKE_AMP)
    self.assertEqual(SINGLB_AMP_RESP['amphora'], ret)