from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import hypervisors as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_use_limit_marker_params(self):
    params = {'limit': '10', 'marker': 'fake-marker'}
    self.cs.hypervisors.list(**params)
    for k, v in params.items():
        self.assertEqual([v], self.requests_mock.last_request.qs[k])