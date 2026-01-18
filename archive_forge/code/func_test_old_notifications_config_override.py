from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.tests import utils as test_utils
def test_old_notifications_config_override(self):
    conf = self.messaging_conf.conf
    conf.set_override('notification_driver', ['messaging'])
    conf.set_override('notification_transport_url', 'http://xyz')
    conf.set_override('notification_topics', ['topic1'])
    self.assertEqual(['messaging'], conf.oslo_messaging_notifications.driver)
    self.assertEqual('http://xyz', conf.oslo_messaging_notifications.transport_url)
    self.assertEqual(['topic1'], conf.oslo_messaging_notifications.topics)
    conf.clear_override('notification_driver')
    conf.clear_override('notification_transport_url')
    conf.clear_override('notification_topics')
    self.assertEqual([], conf.oslo_messaging_notifications.driver)
    self.assertIsNone(conf.oslo_messaging_notifications.transport_url)
    self.assertEqual(['notifications'], conf.oslo_messaging_notifications.topics)