from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import services
class ServicesV2_53TestCase(ServicesV211TestCase):
    api_version = '2.53'

    def _update_body(self, status=None, disabled_reason=None, force_down=None):
        body = {}
        if status is not None:
            body['status'] = status
        if disabled_reason is not None:
            body['disabled_reason'] = disabled_reason
        if force_down is not None:
            body['forced_down'] = force_down
        return body

    def test_services_enable(self):
        service = self.cs.services.enable(fakes.FAKE_SERVICE_UUID_1)
        self.assert_request_id(service, fakes.FAKE_REQUEST_ID_LIST)
        values = self._update_body(status='enabled')
        self.cs.assert_called('PUT', '/os-services/%s' % fakes.FAKE_SERVICE_UUID_1, values)
        self.assertIsInstance(service, self._get_service_type())
        self.assertEqual('enabled', service.status)

    def test_services_delete(self):
        ret = self.cs.services.delete(fakes.FAKE_SERVICE_UUID_1)
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('DELETE', '/os-services/%s' % fakes.FAKE_SERVICE_UUID_1)

    def test_services_disable(self):
        service = self.cs.services.disable(fakes.FAKE_SERVICE_UUID_1)
        self.assert_request_id(service, fakes.FAKE_REQUEST_ID_LIST)
        values = self._update_body(status='disabled')
        self.cs.assert_called('PUT', '/os-services/%s' % fakes.FAKE_SERVICE_UUID_1, values)
        self.assertIsInstance(service, self._get_service_type())
        self.assertEqual('disabled', service.status)

    def test_services_disable_log_reason(self):
        service = self.cs.services.disable_log_reason(fakes.FAKE_SERVICE_UUID_1, 'disable bad host')
        self.assert_request_id(service, fakes.FAKE_REQUEST_ID_LIST)
        values = self._update_body(status='disabled', disabled_reason='disable bad host')
        self.cs.assert_called('PUT', '/os-services/%s' % fakes.FAKE_SERVICE_UUID_1, values)
        self.assertIsInstance(service, self._get_service_type())
        self.assertEqual('disabled', service.status)
        self.assertEqual('disable bad host', service.disabled_reason)

    def test_services_force_down(self):
        service = self.cs.services.force_down(fakes.FAKE_SERVICE_UUID_1, False)
        self.assert_request_id(service, fakes.FAKE_REQUEST_ID_LIST)
        values = self._update_body(force_down=False)
        self.cs.assert_called('PUT', '/os-services/%s' % fakes.FAKE_SERVICE_UUID_1, values)
        self.assertIsInstance(service, self._get_service_type())
        self.assertFalse(service.forced_down)