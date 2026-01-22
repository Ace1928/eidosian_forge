import fixtures
from unittest import mock
from oslo_config import cfg
from stevedore import driver
import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
class GetTransportSadPathTestCase(test_utils.BaseTestCase):
    scenarios = [('invalid_transport_url', dict(url=None, transport_url='invalid', ex=dict(cls=oslo_messaging.InvalidTransportURL, msg_contains='No scheme specified', url='invalid'))), ('invalid_url_param', dict(url='invalid', transport_url=None, ex=dict(cls=oslo_messaging.InvalidTransportURL, msg_contains='No scheme specified', url='invalid'))), ('driver_load_failure', dict(url=None, transport_url='testbackend:/', ex=dict(cls=oslo_messaging.DriverLoadFailure, msg_contains='Failed to load', driver='testbackend')))]

    def test_get_transport_sad(self):
        self.config(transport_url=self.transport_url)
        ex_cls = self.ex.pop('cls')
        ex_msg_contains = self.ex.pop('msg_contains')
        ex = self.assertRaises(ex_cls, oslo_messaging.get_transport, self.conf, url=self.url)
        self.assertIn(ex_msg_contains, str(ex))
        for k, v in self.ex.items():
            self.assertTrue(hasattr(ex, k))
            self.assertEqual(v, str(getattr(ex, k)))