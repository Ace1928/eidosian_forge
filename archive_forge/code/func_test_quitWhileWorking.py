import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
def test_quitWhileWorking(self):
    """
        If L{LockWorker.quit} is invoked during a call to L{LockWorker.do}, all
        recursive work scheduled with L{LockWorker.do} will be completed and
        the lock will be released.
        """
    lock = FakeLock()
    ref = weakref.ref(lock)
    worker = LockWorker(lock, local())

    def phase1():
        worker.do(phase2)
        worker.quit()
        self.assertRaises(AlreadyQuit, worker.do, list)
        phase1.complete = True
    phase1.complete = False

    def phase2():
        phase2.complete = True
        phase2.acquired = lock.acquired
    phase2.complete = False
    worker.do(phase1)
    self.assertEqual(phase1.complete, True)
    self.assertEqual(phase2.complete, True)
    self.assertEqual(lock.acquired, False)
    lock = None
    gc.collect()
    self.assertIs(ref(), None)