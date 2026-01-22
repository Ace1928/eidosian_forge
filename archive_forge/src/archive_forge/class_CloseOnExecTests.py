import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
class CloseOnExecTests(unittest.SynchronousTestCase):
    """
    Tests for L{fdesc._setCloseOnExec} and L{fdesc._unsetCloseOnExec}.
    """
    program = "\nimport os, errno\ntry:\n    os.write(%d, b'lul')\nexcept OSError as e:\n    if e.errno == errno.EBADF:\n        os._exit(0)\n    os._exit(5)\nexcept BaseException:\n    os._exit(10)\nelse:\n    os._exit(20)\n"

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

    def test_setCloseOnExec(self):
        """
        A file descriptor passed to L{fdesc._setCloseOnExec} is not inherited
        by a new process image created with one of the exec family of
        functions.
        """
        with open(self.mktemp(), 'wb') as fObj:
            fdesc._setCloseOnExec(fObj.fileno())
            status = self._execWithFileDescriptor(fObj)
            self.assertTrue(os.WIFEXITED(status))
            self.assertEqual(os.WEXITSTATUS(status), 0)

    def test_unsetCloseOnExec(self):
        """
        A file descriptor passed to L{fdesc._unsetCloseOnExec} is inherited by
        a new process image created with one of the exec family of functions.
        """
        with open(self.mktemp(), 'wb') as fObj:
            fdesc._setCloseOnExec(fObj.fileno())
            fdesc._unsetCloseOnExec(fObj.fileno())
            status = self._execWithFileDescriptor(fObj)
            self.assertTrue(os.WIFEXITED(status))
            self.assertEqual(os.WEXITSTATUS(status), 20)