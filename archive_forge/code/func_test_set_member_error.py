from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_set_member_error(self):
    self.requests_mock.register_uri('PUT', FAKE_LBAAS_URL + 'pools/' + FAKE_PO + '/members/' + FAKE_ME, text='{"faultstring": "%s"}' % self._error_message, status_code=400)
    self.assertRaisesRegex(exceptions.OctaviaClientException, self._error_message, self.api.member_set, pool_id=FAKE_PO, member_id=FAKE_ME, json=SINGLE_ME_UPDATE)