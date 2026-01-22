import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
def _execWithFileDescriptor(self, fObj):
    pid = os.fork()
    if pid == 0:
        try:
            os.execv(sys.executable, [sys.executable, '-c', self.program % (fObj.fileno(),)])
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(30)
    else:
        return untilConcludes(os.waitpid, pid, 0)[1]