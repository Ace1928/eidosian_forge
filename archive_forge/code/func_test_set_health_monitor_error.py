from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_set_health_monitor_error(self):
    self.requests_mock.register_uri('PUT', FAKE_LBAAS_URL + 'healthmonitors/' + FAKE_HM, text='{"faultstring": "%s"}' % self._error_message, status_code=400)
    self.assertRaisesRegex(exceptions.OctaviaClientException, self._error_message, self.api.health_monitor_set, FAKE_HM, json=SINGLE_HM_UPDATE)