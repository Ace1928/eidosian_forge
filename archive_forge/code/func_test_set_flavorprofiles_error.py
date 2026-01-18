from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_set_flavorprofiles_error(self):
    self.requests_mock.register_uri('PUT', FAKE_LBAAS_URL + 'flavorprofiles/' + FAKE_FVPF, text='{"faultstring": "%s"}' % self._error_message, status_code=400)
    self.assertRaisesRegex(exceptions.OctaviaClientException, self._error_message, self.api.flavorprofile_set, FAKE_FVPF, json=SINGLE_FVPF_UPDATE)