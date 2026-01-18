import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def test_selectLast(self):
    """
        L{FDDetector._getImplementation} returns the last method from its
        C{_implementations} list if none of the implementations manage to return
        results which reflect a newly opened file descriptor.
        """

    def failWithWrongResults():
        return [3, 5, 9]

    def failWithOtherWrongResults():
        return [0, 1, 2]
    self.detector._implementations = [failWithWrongResults, failWithOtherWrongResults]
    self.assertIs(failWithOtherWrongResults, self.detector._getImplementation())