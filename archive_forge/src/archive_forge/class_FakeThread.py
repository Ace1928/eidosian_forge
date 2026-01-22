import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
class FakeThread:
    """
    A fake L{threading.Thread}.

    @ivar target: A target function to run.
    @type target: L{callable}

    @ivar started: Has this thread been started?
    @type started: L{bool}
    """

    def __init__(self, target):
        """
        Create a L{FakeThread} with a target.
        """
        self.target = target
        self.started = False

    def start(self):
        """
        Set the "started" flag.
        """
        self.started = True