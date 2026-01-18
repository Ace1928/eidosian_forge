import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def test_selectFirstWorking(self):
    """
        L{FDDetector._getImplementation} returns the first method from its
        C{_implementations} list which returns results which reflect a newly
        opened file descriptor.
        """

    def failWithException():
        raise ValueError('This does not work')

    def failWithWrongResults():
        return [0, 1, 2]

    def correct():
        return self._files[:]
    self.detector._implementations = [failWithException, failWithWrongResults, correct]
    self.assertIs(correct, self.detector._getImplementation())