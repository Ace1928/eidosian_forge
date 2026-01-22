import copy
import pickle
from twisted.persisted.styles import _UniversalPicklingError, unpickleMethod
from twisted.trial import unittest
class UniversalPicklingErrorTests(unittest.TestCase):
    """
    Tests the L{_UniversalPicklingError} exception.
    """

    def raise_UniversalPicklingError(self):
        """
        Raise L{UniversalPicklingError}.
        """
        raise _UniversalPicklingError

    def test_handledByPickleModule(self) -> None:
        """
        Handling L{pickle.PicklingError} handles
        L{_UniversalPicklingError}.
        """
        self.assertRaises(pickle.PicklingError, self.raise_UniversalPicklingError)