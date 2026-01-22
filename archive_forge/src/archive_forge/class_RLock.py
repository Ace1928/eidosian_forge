import threading
import sys
import tempfile
import time
from . import context
from . import process
from . import util
class RLock(SemLock):

    def __init__(self, *, ctx):
        SemLock.__init__(self, RECURSIVE_MUTEX, 1, 1, ctx=ctx)

    def __repr__(self):
        try:
            if self._semlock._is_mine():
                name = process.current_process().name
                if threading.current_thread().name != 'MainThread':
                    name += '|' + threading.current_thread().name
                count = self._semlock._count()
            elif self._semlock._get_value() == 1:
                name, count = ('None', 0)
            elif self._semlock._count() > 0:
                name, count = ('SomeOtherThread', 'nonzero')
            else:
                name, count = ('SomeOtherProcess', 'nonzero')
        except Exception:
            name, count = ('unknown', 'unknown')
        return '<%s(%s, %s)>' % (self.__class__.__name__, name, count)