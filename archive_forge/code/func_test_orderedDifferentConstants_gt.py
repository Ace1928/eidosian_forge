import warnings
from twisted.trial.unittest import TestCase
def test_orderedDifferentConstants_gt(self):
    """
        L{twisted.python.constants._Constant.__gt__} returns C{NotImplemented}
        when comparing constants of different types.
        """
    self.assertEqual(NotImplemented, NamedLetters.alpha.__gt__(ValuedLetters.alpha))