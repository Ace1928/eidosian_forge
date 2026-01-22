from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import services
class ServicesTest(utils.TestCase):
    """Tests for v3.0 behavior"""

    def test_list_services(self):
        svs = cs.services.list()
        cs.assert_called('GET', '/os-services')
        self.assertEqual(3, len(svs))
        for service in svs:
            self.assertIsInstance(service, services.Service)
            self.assertFalse(hasattr(service, 'cluster'))
        self._assert_request_id(svs)

    def test_list_services_with_hostname(self):
        svs = cs.services.list(host='host2')
        cs.assert_called('GET', '/os-services?host=host2')
        self.assertEqual(2, len(svs))
        [self.assertIsInstance(s, services.Service) for s in svs]
        [self.assertEqual('host2', s.host) for s in svs]
        self._assert_request_id(svs)

    def test_list_services_with_binary(self):
        svs = cs.services.list(binary='cinder-volume')
        cs.assert_called('GET', '/os-services?binary=cinder-volume')
        self.assertEqual(2, len(svs))
        [self.assertIsInstance(s, services.Service) for s in svs]
        [self.assertEqual('cinder-volume', s.binary) for s in svs]
        self._assert_request_id(svs)

    def test_list_services_with_host_binary(self):
        svs = cs.services.list('host2', 'cinder-volume')
        cs.assert_called('GET', '/os-services?host=host2&binary=cinder-volume')
        self.assertEqual(1, len(svs))
        [self.assertIsInstance(s, services.Service) for s in svs]
        [self.assertEqual('host2', s.host) for s in svs]
        [self.assertEqual('cinder-volume', s.binary) for s in svs]
        self._assert_request_id(svs)

    def test_services_enable(self):
        s = cs.services.enable('host1', 'cinder-volume')
        values = {'host': 'host1', 'binary': 'cinder-volume'}
        cs.assert_called('PUT', '/os-services/enable', values)
        self.assertIsInstance(s, services.Service)
        self.assertEqual('enabled', s.status)
        self._assert_request_id(s)

    def test_services_disable(self):
        s = cs.services.disable('host1', 'cinder-volume')
        values = {'host': 'host1', 'binary': 'cinder-volume'}
        cs.assert_called('PUT', '/os-services/disable', values)
        self.assertIsInstance(s, services.Service)
        self.assertEqual('disabled', s.status)
        self._assert_request_id(s)

    def test_services_disable_log_reason(self):
        s = cs.services.disable_log_reason('host1', 'cinder-volume', 'disable bad host')
        values = {'host': 'host1', 'binary': 'cinder-volume', 'disabled_reason': 'disable bad host'}
        cs.assert_called('PUT', '/os-services/disable-log-reason', values)
        self.assertIsInstance(s, services.Service)
        self.assertEqual('disabled', s.status)
        self._assert_request_id(s)