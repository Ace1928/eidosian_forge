import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
def test_quitPreventsFutureCalls(self):
    """
        L{ThreadWorker.quit} causes future calls to L{ThreadWorker.do} and
        L{ThreadWorker.quit} to raise L{AlreadyQuit}.
        """
    self.worker.quit()
    self.assertRaises(AlreadyQuit, self.worker.quit)
    self.assertRaises(AlreadyQuit, self.worker.do, list)