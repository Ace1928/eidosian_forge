from zope.interface.verify import verifyObject
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, createMemoryWorker
def test_performWhenNothingToDoYet(self) -> None:
    """
        The C{perform} callable returned by L{createMemoryWorker} will return
        no result when there's no work to do yet.  Since there is no work to
        do, the performer returns C{False}.
        """
    worker, performer = createMemoryWorker()
    self.assertEqual(performer(), False)