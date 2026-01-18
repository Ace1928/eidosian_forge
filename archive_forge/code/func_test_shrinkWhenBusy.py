from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def test_shrinkWhenBusy(self):
    """
        L{Team.shrink} will wait for busy workers to finish being busy and then
        quit them.
        """
    for x in range(10):
        self.team.do(list)
    self.coordinate()
    self.assertEqual(len(self.allUnquitWorkers), 10)
    self.team.shrink(7)
    self.performAllOutstandingWork()
    self.assertEqual(len(self.allUnquitWorkers), 3)