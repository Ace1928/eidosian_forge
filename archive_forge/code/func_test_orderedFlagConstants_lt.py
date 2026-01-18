import warnings
from twisted.trial.unittest import TestCase
def test_orderedFlagConstants_lt(self):
    """
        L{twisted.python.constants.FlagConstant} preserves definition
        order in C{<} comparisons.
        """
    self.assertTrue(PizzaToppings.mozzarella < PizzaToppings.pesto)
    self.assertTrue(PizzaToppings.pesto < PizzaToppings.pepperoni)