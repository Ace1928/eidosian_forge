import warnings
from twisted.trial.unittest import TestCase
def test_orderedNameConstants_lt(self):
    """
        L{twisted.python.constants.NamedConstant} preserves definition
        order in C{<} comparisons.
        """
    self.assertTrue(NamedLetters.alpha < NamedLetters.beta)