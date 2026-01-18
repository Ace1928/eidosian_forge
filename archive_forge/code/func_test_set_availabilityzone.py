from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_set_availabilityzone(self):
    self.requests_mock.register_uri('PUT', FAKE_LBAAS_URL + 'availabilityzones/' + FAKE_AZ, json=SINGLE_AZ_UPDATE, status_code=200)
    ret = self.api.availabilityzone_set(FAKE_AZ, json=SINGLE_AZ_UPDATE)
    self.assertEqual(SINGLE_AZ_UPDATE, ret)