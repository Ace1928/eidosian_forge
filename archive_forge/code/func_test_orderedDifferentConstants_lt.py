import warnings
from twisted.trial.unittest import TestCase
def test_orderedDifferentConstants_lt(self):
    """
        L{twisted.python.constants._Constant.__lt__} returns C{NotImplemented}
        when comparing constants of different types.
        """
    self.assertEqual(NotImplemented, NamedLetters.alpha.__lt__(ValuedLetters.alpha))