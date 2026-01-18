import warnings
from twisted.trial.unittest import TestCase
def test_orderedNameConstants_ge(self):
    """
        L{twisted.python.constants.NamedConstant} preserves definition
        order in C{>=} comparisons.
        """
    self.assertTrue(NamedLetters.alpha >= NamedLetters.alpha)
    self.assertTrue(NamedLetters.beta >= NamedLetters.alpha)