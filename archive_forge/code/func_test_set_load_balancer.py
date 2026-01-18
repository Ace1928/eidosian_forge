from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_set_load_balancer(self):
    self.requests_mock.register_uri('PUT', FAKE_LBAAS_URL + 'loadbalancers/' + FAKE_LB, json=SINGLE_LB_UPDATE, status_code=200)
    ret = self.api.load_balancer_set(FAKE_LB, json=SINGLE_LB_UPDATE)
    self.assertEqual(SINGLE_LB_UPDATE, ret)