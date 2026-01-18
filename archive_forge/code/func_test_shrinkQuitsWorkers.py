from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def test_shrinkQuitsWorkers(self):
    """
        L{Team.shrink} will quit the given number of workers.
        """
    self.team.grow(5)
    self.performAllOutstandingWork()
    self.team.shrink(3)
    self.performAllOutstandingWork()
    self.assertEqual(len(self.allUnquitWorkers), 2)