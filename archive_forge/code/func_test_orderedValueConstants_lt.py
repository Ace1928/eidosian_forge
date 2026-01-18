import warnings
from twisted.trial.unittest import TestCase
def test_orderedValueConstants_lt(self):
    """
        L{twisted.python.constants.ValueConstant} preserves definition
        order in C{<} comparisons.
        """
    self.assertTrue(ValuedLetters.alpha < ValuedLetters.digamma)
    self.assertTrue(ValuedLetters.digamma < ValuedLetters.zeta)