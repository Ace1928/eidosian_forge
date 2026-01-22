import os
import tempfile
import eventlet
from eventlet import greenpool
from oslotest import base as test_base
from oslo_concurrency import lockutils
class TestFileLocks(test_base.BaseTestCase):

    def test_concurrent_green_lock_succeeds(self):
        """Verify spawn_n greenthreads with two locks run concurrently."""
        tmpdir = tempfile.mkdtemp()
        self.completed = False

        def locka(wait):
            a = lockutils.InterProcessLock(os.path.join(tmpdir, 'a'))
            with a:
                wait.wait()
            self.completed = True

        def lockb(wait):
            b = lockutils.InterProcessLock(os.path.join(tmpdir, 'b'))
            with b:
                wait.wait()
        wait1 = eventlet.event.Event()
        wait2 = eventlet.event.Event()
        pool = greenpool.GreenPool()
        pool.spawn_n(locka, wait1)
        pool.spawn_n(lockb, wait2)
        wait2.send()
        eventlet.sleep(0)
        wait1.send()
        pool.waitall()
        self.assertTrue(self.completed)