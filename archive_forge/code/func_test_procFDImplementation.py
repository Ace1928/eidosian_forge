import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def test_procFDImplementation(self):
    """
        L{_FDDetector._procFDImplementation} raises L{OSError} if there is no
        I{/proc/<pid>/fd} directory, otherwise it returns the basenames of its
        children interpreted as integers.
        """
    self.procfs = False
    self.assertRaises(OSError, self.detector._procFDImplementation)
    self.procfs = True
    self.assertEqual([0, 1, 2], self.detector._procFDImplementation())