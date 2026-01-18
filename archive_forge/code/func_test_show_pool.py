from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_show_pool(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'pools/' + FAKE_PO, json=SINGLE_PO_RESP, status_code=200)
    ret = self.api.pool_show(FAKE_PO)
    self.assertEqual(SINGLE_PO_RESP['pool'], ret)