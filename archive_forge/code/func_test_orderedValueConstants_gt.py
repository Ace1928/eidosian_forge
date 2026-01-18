import warnings
from twisted.trial.unittest import TestCase
def test_orderedValueConstants_gt(self):
    """
        L{twisted.python.constants.ValueConstant} preserves definition
        order in C{>} comparisons.
        """
    self.assertTrue(ValuedLetters.digamma > ValuedLetters.alpha)
    self.assertTrue(ValuedLetters.zeta > ValuedLetters.digamma)