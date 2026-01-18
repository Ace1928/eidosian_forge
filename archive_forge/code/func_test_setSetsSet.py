from twisted.trial.unittest import SynchronousTestCase
from .._convenience import Quit
from .._ithreads import AlreadyQuit
def test_setSetsSet(self) -> None:
    """
        L{Quit.set} sets L{Quit.isSet} to L{True}.
        """
    quit = Quit()
    quit.set()
    self.assertEqual(quit.isSet, True)