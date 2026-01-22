import warnings
from twisted.trial.unittest import TestCase
class FlagsTests(_FlagsTestsMixin, TestCase, _ConstantsTestsMixin):
    """
    Tests for L{twisted.python.constants.Flags}, a base class for containers of
    related, combinable flag or bitvector-like constants.
    """

    def test_notInstantiable(self):
        """
        A subclass of L{Flags} raises L{TypeError} if an attempt is made to
        instantiate it.
        """
        self._notInstantiableTest('FXF', self.FXF)

    def test_symbolicAttributes(self):
        """
        Each name associated with a L{FlagConstant} instance in the definition
        of a L{Flags} subclass is available as an attribute on the resulting
        class.
        """
        self.assertTrue(hasattr(self.FXF, 'READ'))
        self.assertTrue(hasattr(self.FXF, 'WRITE'))
        self.assertTrue(hasattr(self.FXF, 'APPEND'))
        self.assertTrue(hasattr(self.FXF, 'EXCLUSIVE'))
        self.assertTrue(hasattr(self.FXF, 'TEXT'))

    def test_withoutOtherAttributes(self):
        """
        As usual, names not defined in the class scope of a L{Flags} subclass
        are not available as attributes on the resulting class.
        """
        self.assertFalse(hasattr(self.FXF, 'foo'))

    def test_representation(self):
        """
        The string representation of a constant on a L{Flags} subclass includes
        the name of the L{Flags} subclass and the name of the constant itself.
        """
        self.assertEqual('<FXF=READ>', repr(self.FXF.READ))

    def test_lookupByName(self):
        """
        Constants can be looked up by name using L{Flags.lookupByName}.
        """
        flag = self.FXF.lookupByName('READ')
        self.assertIs(self.FXF.READ, flag)

    def test_notLookupMissingByName(self):
        """
        Names not defined with a L{FlagConstant} instance cannot be looked up
        using L{Flags.lookupByName}.
        """
        self.assertRaises(ValueError, self.FXF.lookupByName, 'lookupByName')
        self.assertRaises(ValueError, self.FXF.lookupByName, '__init__')
        self.assertRaises(ValueError, self.FXF.lookupByName, 'foo')

    def test_lookupByValue(self):
        """
        Constants can be looked up by their associated value, defined
        implicitly by the position in which the constant appears in the class
        definition or explicitly by the argument passed to L{FlagConstant}.
        """
        flag = self.FXF.lookupByValue(1)
        self.assertIs(flag, self.FXF.READ)
        flag = self.FXF.lookupByValue(2)
        self.assertIs(flag, self.FXF.WRITE)
        flag = self.FXF.lookupByValue(4)
        self.assertIs(flag, self.FXF.APPEND)
        flag = self.FXF.lookupByValue(32)
        self.assertIs(flag, self.FXF.EXCLUSIVE)
        flag = self.FXF.lookupByValue(64)
        self.assertIs(flag, self.FXF.TEXT)

    def test_lookupDuplicateByValue(self):
        """
        If more than one constant is associated with a particular value,
        L{Flags.lookupByValue} returns whichever of them is defined first.
        """

        class TIMEX(Flags):
            ADJ_OFFSET = FlagConstant(1)
            MOD_OFFSET = FlagConstant(1)
        self.assertIs(TIMEX.lookupByValue(1), TIMEX.ADJ_OFFSET)

    def test_notLookupMissingByValue(self):
        """
        L{Flags.lookupByValue} raises L{ValueError} when called with a value
        with which no constant is associated.
        """
        self.assertRaises(ValueError, self.FXF.lookupByValue, 16)

    def test_name(self):
        """
        The C{name} attribute of one of the constants gives that constant's
        name.
        """
        self.assertEqual('READ', self.FXF.READ.name)

    def test_attributeIdentity(self):
        """
        Repeated access of an attribute associated with a L{FlagConstant} value
        in a L{Flags} subclass results in the same object.
        """
        self.assertIs(self.FXF.READ, self.FXF.READ)

    def test_iterconstants(self):
        """
        L{Flags.iterconstants} returns an iterator over all of the constants
        defined in the class, in the order they were defined.
        """
        constants = list(self.FXF.iterconstants())
        self.assertEqual([self.FXF.READ, self.FXF.WRITE, self.FXF.APPEND, self.FXF.EXCLUSIVE, self.FXF.TEXT], constants)

    def test_attributeIterconstantsIdentity(self):
        """
        The constants returned from L{Flags.iterconstants} are identical to the
        constants accessible using attributes.
        """
        constants = list(self.FXF.iterconstants())
        self.assertIs(self.FXF.READ, constants[0])
        self.assertIs(self.FXF.WRITE, constants[1])
        self.assertIs(self.FXF.APPEND, constants[2])
        self.assertIs(self.FXF.EXCLUSIVE, constants[3])
        self.assertIs(self.FXF.TEXT, constants[4])

    def test_iterconstantsIdentity(self):
        """
        The constants returned from L{Flags.iterconstants} are identical on
        each call to that method.
        """
        constants = list(self.FXF.iterconstants())
        again = list(self.FXF.iterconstants())
        self.assertIs(again[0], constants[0])
        self.assertIs(again[1], constants[1])
        self.assertIs(again[2], constants[2])
        self.assertIs(again[3], constants[3])
        self.assertIs(again[4], constants[4])

    def test_initializedOnce(self):
        """
        L{Flags._enumerants} is initialized once and its value re-used on
        subsequent access.
        """
        self._initializedOnceTest(self.FXF, 'READ')