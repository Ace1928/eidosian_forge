import testscenarios
from unittest import mock
from confluent_kafka import KafkaException
import oslo_messaging
from oslo_messaging._drivers import impl_kafka as kafka_driver
from oslo_messaging.tests import utils as test_utils
class TestKafkaConnection(test_utils.BaseTestCase):

    def setUp(self):
        super(TestKafkaConnection, self).setUp()
        self.messaging_conf.transport_url = 'kafka:/'
        transport = oslo_messaging.get_notification_transport(self.conf)
        self.driver = transport._driver

    def test_notify(self):
        with mock.patch('confluent_kafka.Producer') as producer:
            self.driver.pconn.notify_send('fake_topic', {'fake_ctxt': 'fake_param'}, {'fake_text': 'fake_message_1'}, 10)
            assert producer.call_count == 1