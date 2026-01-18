import warnings
from twisted.trial.unittest import TestCase
def test_orderedValueConstants_ge(self):
    """
        L{twisted.python.constants.ValueConstant} preserves definition
        order in C{>=} comparisons.
        """
    self.assertTrue(ValuedLetters.alpha >= ValuedLetters.alpha)
    self.assertTrue(ValuedLetters.digamma >= ValuedLetters.alpha)
    self.assertTrue(ValuedLetters.zeta >= ValuedLetters.digamma)