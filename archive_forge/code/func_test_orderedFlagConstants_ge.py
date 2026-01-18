import warnings
from twisted.trial.unittest import TestCase
def test_orderedFlagConstants_ge(self):
    """
        L{twisted.python.constants.FlagConstant} preserves definition
        order in C{>=} comparisons.
        """
    self.assertTrue(PizzaToppings.mozzarella >= PizzaToppings.mozzarella)
    self.assertTrue(PizzaToppings.pesto >= PizzaToppings.mozzarella)
    self.assertTrue(PizzaToppings.pepperoni >= PizzaToppings.pesto)