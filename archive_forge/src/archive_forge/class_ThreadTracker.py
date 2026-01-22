import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class ThreadTracker(object):

    def __init__(self):
        self._received_msgs = 0
        self.threads = []
        self.lock = threading.Condition()

    def info(self, *args, **kwargs):
        with self.lock:
            self._received_msgs += 1
            self.lock.notify_all()

    def wait_for_messages(self, expect_messages):
        with self.lock:
            while self._received_msgs < expect_messages:
                self.lock.wait()

    def stop(self):
        for thread in self.threads:
            thread.stop()
        self.threads = []

    def start(self, thread):
        self.threads.append(thread)
        thread.start()