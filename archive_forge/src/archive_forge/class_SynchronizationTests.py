import pickle
import sys
from unittest import skipIf
from twisted.python import threadable
from twisted.trial.unittest import FailTest, SynchronousTestCase
class SynchronizationTests(SynchronousTestCase):

    def setUp(self):
        """
        Reduce the CPython check interval so that thread switches happen much
        more often, hopefully exercising more possible race conditions.  Also,
        delay actual test startup until the reactor has been started.
        """
        self.addCleanup(sys.setswitchinterval, sys.getswitchinterval())
        sys.setswitchinterval(1e-07)

    def test_synchronizedName(self):
        """
        The name of a synchronized method is inaffected by the synchronization
        decorator.
        """
        self.assertEqual('aMethod', TestObject.aMethod.__name__)

    @skipIf(threadingSkip, 'Platform does not support threads')
    def test_isInIOThread(self):
        """
        L{threadable.isInIOThread} returns C{True} if and only if it is called
        in the same thread as L{threadable.registerAsIOThread}.
        """
        threadable.registerAsIOThread()
        foreignResult = []
        t = threading.Thread(target=lambda: foreignResult.append(threadable.isInIOThread()))
        t.start()
        t.join()
        self.assertFalse(foreignResult[0], 'Non-IO thread reported as IO thread')
        self.assertTrue(threadable.isInIOThread(), 'IO thread reported as not IO thread')

    @skipIf(threadingSkip, 'Platform does not support threads')
    def testThreadedSynchronization(self):
        o = TestObject()
        errors = []

        def callMethodLots():
            try:
                for i in range(1000):
                    o.aMethod()
            except AssertionError as e:
                errors.append(str(e))
        threads = []
        for x in range(5):
            t = threading.Thread(target=callMethodLots)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise FailTest(errors)

    def testUnthreadedSynchronization(self):
        o = TestObject()
        for i in range(1000):
            o.aMethod()