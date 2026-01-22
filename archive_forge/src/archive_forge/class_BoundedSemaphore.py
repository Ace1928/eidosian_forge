import threading
import sys
import tempfile
import time
from . import context
from . import process
from . import util
class BoundedSemaphore(Semaphore):

    def __init__(self, value=1, *, ctx):
        SemLock.__init__(self, SEMAPHORE, value, value, ctx=ctx)

    def __repr__(self):
        try:
            value = self._semlock._get_value()
        except Exception:
            value = 'unknown'
        return '<%s(value=%s, maxvalue=%s)>' % (self.__class__.__name__, value, self._semlock.maxvalue)