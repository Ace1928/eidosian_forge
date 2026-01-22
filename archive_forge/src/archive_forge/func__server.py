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
def _server(self, target):
    ctrl = None
    if self.use_fanout_ctrl:
        ctrl = self._target(fanout=True)
    server = RpcServerFixture(self.conf, self.url, target, endpoint=self.endpoint, ctrl_target=ctrl)
    return server