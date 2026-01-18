from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_set_flavorprofiles(self):
    self.requests_mock.register_uri('PUT', FAKE_LBAAS_URL + 'flavorprofiles/' + FAKE_FVPF, json=SINGLE_FVPF_UPDATE, status_code=200)
    ret = self.api.flavorprofile_set(FAKE_FVPF, json=SINGLE_FVPF_UPDATE)
    self.assertEqual(SINGLE_FVPF_UPDATE, ret)