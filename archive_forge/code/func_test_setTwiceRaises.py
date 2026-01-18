from twisted.trial.unittest import SynchronousTestCase
from .._convenience import Quit
from .._ithreads import AlreadyQuit
def test_setTwiceRaises(self) -> None:
    """
        L{Quit.set} raises L{AlreadyQuit} if it has been called previously.
        """
    quit = Quit()
    quit.set()
    self.assertRaises(AlreadyQuit, quit.set)