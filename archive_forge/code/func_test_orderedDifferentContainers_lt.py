import warnings
from twisted.trial.unittest import TestCase
def test_orderedDifferentContainers_lt(self):
    """
        L{twisted.python.constants._Constant.__lt__} returns C{NotImplemented}
        when comparing constants belonging to different containers.
        """
    self.assertEqual(NotImplemented, NamedLetters.alpha.__lt__(MoreNamedLetters.digamma))