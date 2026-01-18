from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import aggregates as data
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import aggregates
from novaclient.v2 import images
def test_update_with_availability_zone(self):
    aggregate = self.cs.aggregates.get('1')
    values = {'name': 'foo', 'availability_zone': 'new_zone'}
    body = {'aggregate': values}
    result3 = self.cs.aggregates.update(aggregate, values)
    self.assert_request_id(result3, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('PUT', '/os-aggregates/1', body)
    self.assertIsInstance(result3, aggregates.Aggregate)