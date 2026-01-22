from _pydevd_bundle.pydevd_constants import ForkSafeLock, get_global_debugger
import os
import sys
from contextlib import contextmanager
class IOBuf:
    """This class works as a replacement for stdio and stderr.
    It is a buffer and when its contents are requested, it will erase what
    it has so far so that the next return will not return the same contents again.
    """

    def __init__(self):
        self.buflist = []
        import os
        self.encoding = os.environ.get('PYTHONIOENCODING', 'utf-8')

    def getvalue(self):
        b = self.buflist
        self.buflist = []
        return ''.join(b)

    def write(self, s):
        if isinstance(s, bytes):
            s = s.decode(self.encoding, errors='replace')
        self.buflist.append(s)

    def isatty(self):
        return False

    def flush(self):
        pass

    def empty(self):
        return len(self.buflist) == 0