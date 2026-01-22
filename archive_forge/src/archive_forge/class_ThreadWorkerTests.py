import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
class ThreadWorkerTests(SynchronousTestCase):
    """
    Tests for L{ThreadWorker}.
    """

    def setUp(self):
        """
        Create a worker with fake threads.
        """
        self.fakeThreads = []
        self.fakeQueue = FakeQueue()

        def startThread(target):
            newThread = FakeThread(target=target)
            newThread.start()
            self.fakeThreads.append(newThread)
            return newThread
        self.worker = ThreadWorker(startThread, self.fakeQueue)

    def test_startsThreadAndPerformsWork(self):
        """
        L{ThreadWorker} calls its C{createThread} callable to create a thread,
        its C{createQueue} callable to create a queue, and then the thread's
        target pulls work from that queue.
        """
        self.assertEqual(len(self.fakeThreads), 1)
        self.assertEqual(self.fakeThreads[0].started, True)

        def doIt():
            doIt.done = True
        doIt.done = False
        self.worker.do(doIt)
        self.assertEqual(doIt.done, False)
        self.assertRaises(FakeQueueEmpty, self.fakeThreads[0].target)
        self.assertEqual(doIt.done, True)

    def test_quitPreventsFutureCalls(self):
        """
        L{ThreadWorker.quit} causes future calls to L{ThreadWorker.do} and
        L{ThreadWorker.quit} to raise L{AlreadyQuit}.
        """
        self.worker.quit()
        self.assertRaises(AlreadyQuit, self.worker.quit)
        self.assertRaises(AlreadyQuit, self.worker.do, list)