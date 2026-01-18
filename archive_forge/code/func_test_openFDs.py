import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def test_openFDs(self):
    """
        File descriptors returned by L{_listOpenFDs} are mostly open.

        This test assumes that zero-legth writes fail with EBADF on closed
        file descriptors.
        """
    for fd in process._listOpenFDs():
        try:
            fcntl.fcntl(fd, fcntl.F_GETFL)
        except OSError as err:
            self.assertEqual(errno.EBADF, err.errno, 'fcntl(%d, F_GETFL) failed with unexpected errno %d' % (fd, err.errno))