import os
import queue
import time
import uuid
import fixtures
from oslo_config import cfg
import oslo_messaging
from oslo_messaging._drivers.kafka_driver import kafka_options
from oslo_messaging.notify import notifier
from oslo_messaging.tests import utils as test_utils
class SkipIfNoTransportURL(test_utils.BaseTestCase):

    def setUp(self, conf=cfg.CONF):
        super(SkipIfNoTransportURL, self).setUp(conf=conf)
        self.rpc_url = os.environ.get('RPC_TRANSPORT_URL')
        self.notify_url = os.environ.get('NOTIFY_TRANSPORT_URL')
        if not (self.rpc_url or self.notify_url):
            self.skipTest('No transport url configured')
        transport_url = oslo_messaging.TransportURL.parse(conf, self.notify_url)
        kafka_options.register_opts(conf, transport_url)