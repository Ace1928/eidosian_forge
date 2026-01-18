from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def test_shrinkToZero(self):
    """
        L{Team.shrink} with no arguments will stop all outstanding workers.
        """
    self.team.grow(10)
    self.performAllOutstandingWork()
    self.assertEqual(len(self.allUnquitWorkers), 10)
    self.team.shrink()
    self.assertEqual(len(self.allUnquitWorkers), 10)
    self.performAllOutstandingWork()
    self.assertEqual(len(self.allUnquitWorkers), 0)