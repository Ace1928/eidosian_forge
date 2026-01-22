import os
import tempfile
import eventlet
from eventlet import greenpool
from oslotest import base as test_base
from oslo_concurrency import lockutils
def _test_internal_lock_with_two_threads(self, fair, spawn):
    self.other_started = eventlet.event.Event()
    self.other_finished = eventlet.event.Event()

    def other():
        self.other_started.send('started')
        with lockutils.lock('my-lock', fair=fair):
            pass
        self.other_finished.send('finished')
    with lockutils.lock('my-lock', fair=fair):
        spawn(other)
        self.other_started.wait()
        eventlet.sleep(0)
        self.assertIsNone(self.other_finished.wait(0.5), 'Two threads was able to take the same lock')
    result = self.other_finished.wait()
    self.assertEqual('finished', result)