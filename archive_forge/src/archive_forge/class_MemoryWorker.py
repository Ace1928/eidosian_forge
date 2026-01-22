from zope.interface import implementer
from . import IWorker
from ._convenience import Quit
@implementer(IWorker)
class MemoryWorker:
    """
    An L{IWorker} that queues work for later performance.

    @ivar _quit: a flag indicating
    @type _quit: L{Quit}
    """

    def __init__(self, pending=list):
        """
        Create a L{MemoryWorker}.
        """
        self._quit = Quit()
        self._pending = pending()

    def do(self, work):
        """
        Queue some work for to perform later; see L{createMemoryWorker}.

        @param work: The work to perform.
        """
        self._quit.check()
        self._pending.append(work)

    def quit(self):
        """
        Quit this worker.
        """
        self._quit.set()
        self._pending.append(NoMoreWork)