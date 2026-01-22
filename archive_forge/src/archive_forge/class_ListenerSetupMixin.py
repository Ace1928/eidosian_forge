import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class ListenerSetupMixin(object):

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

    def setUp(self):
        self.trackers = {}
        self.addCleanup(self._stop_trackers)

    def _stop_trackers(self):
        for pool in self.trackers:
            self.trackers[pool].stop()
        self.trackers = {}

    def _setup_listener(self, transport, endpoints, targets=None, pool=None, batch=False):
        if pool is None:
            tracker_name = '__default__'
        else:
            tracker_name = pool
        if targets is None:
            targets = [oslo_messaging.Target(topic='testtopic')]
        tracker = self.trackers.setdefault(tracker_name, self.ThreadTracker())
        if batch:
            listener = oslo_messaging.get_batch_notification_listener(transport, targets=targets, endpoints=[tracker] + endpoints, allow_requeue=True, pool=pool, executor='eventlet', batch_size=batch[0], batch_timeout=batch[1])
        else:
            listener = oslo_messaging.get_notification_listener(transport, targets=targets, endpoints=[tracker] + endpoints, allow_requeue=True, pool=pool, executor='eventlet')
        thread = RestartableServerThread(listener)
        tracker.start(thread)
        return thread

    def wait_for_messages(self, expect_messages, tracker_name='__default__'):
        self.trackers[tracker_name].wait_for_messages(expect_messages)

    def _setup_notifier(self, transport, topics=['testtopic'], publisher_id='testpublisher'):
        return oslo_messaging.Notifier(transport, topics=topics, driver='messaging', publisher_id=publisher_id)