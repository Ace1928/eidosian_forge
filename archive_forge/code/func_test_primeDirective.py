import copy
import pickle
from twisted.persisted.styles import _UniversalPicklingError, unpickleMethod
from twisted.trial import unittest
def test_primeDirective(self) -> None:
    """
        We do not contaminate normal function pickling with concerns from
        Twisted.
        """

    def expected(n):
        return '\n'.join(['c' + __name__, sampleFunction.__name__, 'p' + n, '.']).encode('ascii')
    self.assertEqual(pickle.dumps(sampleFunction, protocol=0), expected('0'))