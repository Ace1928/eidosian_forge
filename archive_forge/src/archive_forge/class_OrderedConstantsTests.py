import warnings
from twisted.trial.unittest import TestCase
class OrderedConstantsTests(TestCase):
    """
    Tests for the ordering of constants.  All constants are ordered by
    the order in which they are defined in their container class.
    The ordering of constants that are not in the same container is not
    defined.
    """

    def test_orderedNameConstants_lt(self):
        """
        L{twisted.python.constants.NamedConstant} preserves definition
        order in C{<} comparisons.
        """
        self.assertTrue(NamedLetters.alpha < NamedLetters.beta)

    def test_orderedNameConstants_le(self):
        """
        L{twisted.python.constants.NamedConstant} preserves definition
        order in C{<=} comparisons.
        """
        self.assertTrue(NamedLetters.alpha <= NamedLetters.alpha)
        self.assertTrue(NamedLetters.alpha <= NamedLetters.beta)

    def test_orderedNameConstants_gt(self):
        """
        L{twisted.python.constants.NamedConstant} preserves definition
        order in C{>} comparisons.
        """
        self.assertTrue(NamedLetters.beta > NamedLetters.alpha)

    def test_orderedNameConstants_ge(self):
        """
        L{twisted.python.constants.NamedConstant} preserves definition
        order in C{>=} comparisons.
        """
        self.assertTrue(NamedLetters.alpha >= NamedLetters.alpha)
        self.assertTrue(NamedLetters.beta >= NamedLetters.alpha)

    def test_orderedValueConstants_lt(self):
        """
        L{twisted.python.constants.ValueConstant} preserves definition
        order in C{<} comparisons.
        """
        self.assertTrue(ValuedLetters.alpha < ValuedLetters.digamma)
        self.assertTrue(ValuedLetters.digamma < ValuedLetters.zeta)

    def test_orderedValueConstants_le(self):
        """
        L{twisted.python.constants.ValueConstant} preserves definition
        order in C{<=} comparisons.
        """
        self.assertTrue(ValuedLetters.alpha <= ValuedLetters.alpha)
        self.assertTrue(ValuedLetters.alpha <= ValuedLetters.digamma)
        self.assertTrue(ValuedLetters.digamma <= ValuedLetters.zeta)

    def test_orderedValueConstants_gt(self):
        """
        L{twisted.python.constants.ValueConstant} preserves definition
        order in C{>} comparisons.
        """
        self.assertTrue(ValuedLetters.digamma > ValuedLetters.alpha)
        self.assertTrue(ValuedLetters.zeta > ValuedLetters.digamma)

    def test_orderedValueConstants_ge(self):
        """
        L{twisted.python.constants.ValueConstant} preserves definition
        order in C{>=} comparisons.
        """
        self.assertTrue(ValuedLetters.alpha >= ValuedLetters.alpha)
        self.assertTrue(ValuedLetters.digamma >= ValuedLetters.alpha)
        self.assertTrue(ValuedLetters.zeta >= ValuedLetters.digamma)

    def test_orderedFlagConstants_lt(self):
        """
        L{twisted.python.constants.FlagConstant} preserves definition
        order in C{<} comparisons.
        """
        self.assertTrue(PizzaToppings.mozzarella < PizzaToppings.pesto)
        self.assertTrue(PizzaToppings.pesto < PizzaToppings.pepperoni)

    def test_orderedFlagConstants_le(self):
        """
        L{twisted.python.constants.FlagConstant} preserves definition
        order in C{<=} comparisons.
        """
        self.assertTrue(PizzaToppings.mozzarella <= PizzaToppings.mozzarella)
        self.assertTrue(PizzaToppings.mozzarella <= PizzaToppings.pesto)
        self.assertTrue(PizzaToppings.pesto <= PizzaToppings.pepperoni)

    def test_orderedFlagConstants_gt(self):
        """
        L{twisted.python.constants.FlagConstant} preserves definition
        order in C{>} comparisons.
        """
        self.assertTrue(PizzaToppings.pesto > PizzaToppings.mozzarella)
        self.assertTrue(PizzaToppings.pepperoni > PizzaToppings.pesto)

    def test_orderedFlagConstants_ge(self):
        """
        L{twisted.python.constants.FlagConstant} preserves definition
        order in C{>=} comparisons.
        """
        self.assertTrue(PizzaToppings.mozzarella >= PizzaToppings.mozzarella)
        self.assertTrue(PizzaToppings.pesto >= PizzaToppings.mozzarella)
        self.assertTrue(PizzaToppings.pepperoni >= PizzaToppings.pesto)

    def test_orderedDifferentConstants_lt(self):
        """
        L{twisted.python.constants._Constant.__lt__} returns C{NotImplemented}
        when comparing constants of different types.
        """
        self.assertEqual(NotImplemented, NamedLetters.alpha.__lt__(ValuedLetters.alpha))

    def test_orderedDifferentConstants_le(self):
        """
        L{twisted.python.constants._Constant.__le__} returns C{NotImplemented}
        when comparing constants of different types.
        """
        self.assertEqual(NotImplemented, NamedLetters.alpha.__le__(ValuedLetters.alpha))

    def test_orderedDifferentConstants_gt(self):
        """
        L{twisted.python.constants._Constant.__gt__} returns C{NotImplemented}
        when comparing constants of different types.
        """
        self.assertEqual(NotImplemented, NamedLetters.alpha.__gt__(ValuedLetters.alpha))

    def test_orderedDifferentConstants_ge(self):
        """
        L{twisted.python.constants._Constant.__ge__} returns C{NotImplemented}
        when comparing constants of different types.
        """
        self.assertEqual(NotImplemented, NamedLetters.alpha.__ge__(ValuedLetters.alpha))

    def test_orderedDifferentContainers_lt(self):
        """
        L{twisted.python.constants._Constant.__lt__} returns C{NotImplemented}
        when comparing constants belonging to different containers.
        """
        self.assertEqual(NotImplemented, NamedLetters.alpha.__lt__(MoreNamedLetters.digamma))

    def test_orderedDifferentContainers_le(self):
        """
        L{twisted.python.constants._Constant.__le__} returns C{NotImplemented}
        when comparing constants belonging to different containers.
        """
        self.assertEqual(NotImplemented, NamedLetters.alpha.__le__(MoreNamedLetters.digamma))

    def test_orderedDifferentContainers_gt(self):
        """
        L{twisted.python.constants._Constant.__gt__} returns C{NotImplemented}
        when comparing constants belonging to different containers.
        """
        self.assertEqual(NotImplemented, NamedLetters.alpha.__gt__(MoreNamedLetters.digamma))

    def test_orderedDifferentContainers_ge(self):
        """
        L{twisted.python.constants._Constant.__ge__} returns C{NotImplemented}
        when comparing constants belonging to different containers.
        """
        self.assertEqual(NotImplemented, NamedLetters.alpha.__ge__(MoreNamedLetters.digamma))