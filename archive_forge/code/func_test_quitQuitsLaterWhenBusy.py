from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def test_quitQuitsLaterWhenBusy(self):
    """
        L{Team.quit} causes all busy workers to be quit once they've finished
        the work they've been given.
        """
    self.team.grow(10)
    for x in range(5):
        self.team.do(list)
    self.coordinate()
    self.team.quit()
    self.coordinate()
    self.assertEqual(len(self.allUnquitWorkers), 5)
    self.performAllOutstandingWork()
    self.assertEqual(len(self.allUnquitWorkers), 0)
    self.assertRaises(AlreadyQuit, self.coordinator.quit)