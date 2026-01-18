from osc_lib import exceptions
from openstackclient.api import api
from openstackclient.tests.unit.api import fakes as api_fakes
def test_session_request(self):
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz', json=api_fakes.RESP_ITEM_1, status_code=200)
    ret = self.api._request('GET', '/qaz')
    self.assertEqual(api_fakes.RESP_ITEM_1, ret.json())