from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_show_flavorprofile(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'flavorprofiles/' + FAKE_FVPF, json=SINGLE_FVPF_RESP, status_code=200)
    ret = self.api.flavorprofile_show(FAKE_FVPF)
    self.assertEqual(SINGLE_FVPF_RESP['flavorprofile'], ret)