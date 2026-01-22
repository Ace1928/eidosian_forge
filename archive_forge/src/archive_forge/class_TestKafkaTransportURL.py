import testscenarios
from unittest import mock
from confluent_kafka import KafkaException
import oslo_messaging
from oslo_messaging._drivers import impl_kafka as kafka_driver
from oslo_messaging.tests import utils as test_utils
class TestKafkaTransportURL(test_utils.BaseTestCase):
    scenarios = [('port', dict(url='kafka://localhost:1234', expected=dict(hostaddrs=['localhost:1234'], username=None, password=None, vhost=None))), ('vhost', dict(url='kafka://localhost:1234/my_host', expected=dict(hostaddrs=['localhost:1234'], username=None, password=None, vhost='my_host'))), ('two', dict(url='kafka://localhost:1234,localhost2:1234', expected=dict(hostaddrs=['localhost:1234', 'localhost2:1234'], username=None, password=None, vhost=None))), ('user', dict(url='kafka://stack:stacksecret@localhost:9092/my_host', expected=dict(hostaddrs=['localhost:9092'], username='stack', password='stacksecret', vhost='my_host'))), ('user2', dict(url='kafka://stack:stacksecret@localhost:9092,stack2:stacksecret2@localhost:1234/my_host', expected=dict(hostaddrs=['localhost:9092', 'localhost:1234'], username='stack', password='stacksecret', vhost='my_host')))]

    def setUp(self):
        super(TestKafkaTransportURL, self).setUp()
        self.messaging_conf.transport_url = 'kafka:/'

    def test_transport_url(self):
        transport = oslo_messaging.get_notification_transport(self.conf, self.url)
        self.addCleanup(transport.cleanup)
        driver = transport._driver
        self.assertIsInstance(driver, kafka_driver.KafkaDriver)
        self.assertEqual(self.expected['hostaddrs'], driver.pconn.hostaddrs)
        self.assertEqual(self.expected['username'], driver.pconn.username)
        self.assertEqual(self.expected['password'], driver.pconn.password)
        self.assertEqual(self.expected['vhost'], driver.virtual_host)