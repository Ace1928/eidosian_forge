import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def wait_for_messages(self, expect_messages):
    with self.lock:
        while self._received_msgs < expect_messages:
            self.lock.wait()