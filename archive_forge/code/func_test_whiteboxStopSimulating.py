from typing import TYPE_CHECKING, List
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
def test_whiteboxStopSimulating(self) -> None:
    """
        CFReactor's simulation timer is None after CFReactor crashes.
        """
    r = self.buildReactor()
    r.callLater(0, r.crash)
    r.callLater(100, noop)
    self.runReactor(r)
    self.assertIs(r._currentSimulator, None)