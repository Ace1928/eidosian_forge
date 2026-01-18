from typing import TYPE_CHECKING, List
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
def test_whiteboxIterate(self) -> None:
    """
        C{.iterate()} should remove the CFTimer that will run Twisted's
        callLaters from the loop, even if one is still pending.  We test this
        state indirectly with a white-box assertion by verifying the
        C{_currentSimulator} is set to C{None}, since CoreFoundation does not
        allow us to enumerate all active timers or sources.
        """
    r = self.buildReactor()
    x: List[int] = []
    r.callLater(0, x.append, 1)
    delayed = r.callLater(100, noop)
    r.iterate()
    self.assertIs(r._currentSimulator, None)
    self.assertEqual(r.getDelayedCalls(), [delayed])
    self.assertEqual(x, [1])