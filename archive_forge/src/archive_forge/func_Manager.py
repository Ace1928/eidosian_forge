import os
import sys
import threading
from . import process
from . import reduction
def Manager(self):
    """Returns a manager associated with a running server process

        The managers methods such as `Lock()`, `Condition()` and `Queue()`
        can be used to create shared objects.
        """
    from .managers import SyncManager
    m = SyncManager(ctx=self.get_context())
    m.start()
    return m