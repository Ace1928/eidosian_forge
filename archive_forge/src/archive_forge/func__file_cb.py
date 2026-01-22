import asyncio
import os
import sys
from eventlet.hubs import hub
from eventlet.patcher import original
def _file_cb(self, cb, fileno):
    """
        Callback called by ``asyncio`` when a file descriptor has an event.
        """
    try:
        cb(fileno)
    except self.SYSTEM_EXCEPTIONS:
        raise
    except:
        self.squelch_exception(fileno, sys.exc_info())
    self.sleep_event.set()