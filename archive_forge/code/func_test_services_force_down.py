from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import services
def test_services_force_down(self):
    service = self.cs.services.force_down(fakes.FAKE_SERVICE_UUID_1, False)
    self.assert_request_id(service, fakes.FAKE_REQUEST_ID_LIST)
    values = self._update_body(force_down=False)
    self.cs.assert_called('PUT', '/os-services/%s' % fakes.FAKE_SERVICE_UUID_1, values)
    self.assertIsInstance(service, self._get_service_type())
    self.assertFalse(service.forced_down)