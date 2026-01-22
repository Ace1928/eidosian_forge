import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
class FakeQueue:
    """
    A fake L{Queue} implementing C{put} and C{get}.

    @ivar items: A lit of items placed by C{put} but not yet retrieved by
        C{get}.
    @type items: L{list}
    """

    def __init__(self):
        """
        Create a L{FakeQueue}.
        """
        self.items = []

    def put(self, item):
        """
        Put an item into the queue for later retrieval by L{FakeQueue.get}.

        @param item: any object
        """
        self.items.append(item)

    def get(self):
        """
        Get an item.

        @return: an item previously put by C{put}.
        """
        if not self.items:
            raise FakeQueueEmpty()
        return self.items.pop(0)