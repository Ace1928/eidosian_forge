import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class EnumTest(test_util.TestCase):

    def setUp(self):
        """Set up tests."""
        global Color

        class Color(messages.Enum):
            RED = 20
            ORANGE = 2
            YELLOW = 40
            GREEN = 4
            BLUE = 50
            INDIGO = 5
            VIOLET = 80

    def testNames(self):
        """Test that names iterates over enum names."""
        self.assertEquals(set(['BLUE', 'GREEN', 'INDIGO', 'ORANGE', 'RED', 'VIOLET', 'YELLOW']), set(Color.names()))

    def testNumbers(self):
        """Tests that numbers iterates of enum numbers."""
        self.assertEquals(set([2, 4, 5, 20, 40, 50, 80]), set(Color.numbers()))

    def testIterate(self):
        """Test that __iter__ iterates over all enum values."""
        self.assertEquals(set(Color), set([Color.RED, Color.ORANGE, Color.YELLOW, Color.GREEN, Color.BLUE, Color.INDIGO, Color.VIOLET]))

    def testNaturalOrder(self):
        """Test that natural order enumeration is in numeric order."""
        self.assertEquals([Color.ORANGE, Color.GREEN, Color.INDIGO, Color.RED, Color.YELLOW, Color.BLUE, Color.VIOLET], sorted(Color))

    def testByName(self):
        """Test look-up by name."""
        self.assertEquals(Color.RED, Color.lookup_by_name('RED'))
        self.assertRaises(KeyError, Color.lookup_by_name, 20)
        self.assertRaises(KeyError, Color.lookup_by_name, Color.RED)

    def testByNumber(self):
        """Test look-up by number."""
        self.assertRaises(KeyError, Color.lookup_by_number, 'RED')
        self.assertEquals(Color.RED, Color.lookup_by_number(20))
        self.assertRaises(KeyError, Color.lookup_by_number, Color.RED)

    def testConstructor(self):
        """Test that constructor look-up by name or number."""
        self.assertEquals(Color.RED, Color('RED'))
        self.assertEquals(Color.RED, Color(u'RED'))
        self.assertEquals(Color.RED, Color(20))
        if six.PY2:
            self.assertEquals(Color.RED, Color(long(20)))
        self.assertEquals(Color.RED, Color(Color.RED))
        self.assertRaises(TypeError, Color, 'Not exists')
        self.assertRaises(TypeError, Color, 'Red')
        self.assertRaises(TypeError, Color, 100)
        self.assertRaises(TypeError, Color, 10.0)

    def testLen(self):
        """Test that len function works to count enums."""
        self.assertEquals(7, len(Color))

    def testNoSubclasses(self):
        """Test that it is not possible to sub-class enum classes."""

        def declare_subclass():

            class MoreColor(Color):
                pass
        self.assertRaises(messages.EnumDefinitionError, declare_subclass)

    def testClassNotMutable(self):
        """Test that enum classes themselves are not mutable."""
        self.assertRaises(AttributeError, setattr, Color, 'something_new', 10)

    def testInstancesMutable(self):
        """Test that enum instances are not mutable."""
        self.assertRaises(TypeError, setattr, Color.RED, 'something_new', 10)

    def testDefEnum(self):
        """Test def_enum works by building enum class from dict."""
        WeekDay = messages.Enum.def_enum({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 6, 'Saturday': 7, 'Sunday': 8}, 'WeekDay')
        self.assertEquals('Wednesday', WeekDay(3).name)
        self.assertEquals(6, WeekDay('Friday').number)
        self.assertEquals(WeekDay.Sunday, WeekDay('Sunday'))

    def testNonInt(self):
        """Test that non-integer values rejection by enum def."""
        self.assertRaises(messages.EnumDefinitionError, messages.Enum.def_enum, {'Bad': '1'}, 'BadEnum')

    def testNegativeInt(self):
        """Test that negative numbers rejection by enum def."""
        self.assertRaises(messages.EnumDefinitionError, messages.Enum.def_enum, {'Bad': -1}, 'BadEnum')

    def testLowerBound(self):
        """Test that zero is accepted by enum def."""

        class NotImportant(messages.Enum):
            """Testing for value zero"""
            VALUE = 0
        self.assertEquals(0, int(NotImportant.VALUE))

    def testTooLargeInt(self):
        """Test that numbers too large are rejected."""
        self.assertRaises(messages.EnumDefinitionError, messages.Enum.def_enum, {'Bad': 2 ** 29}, 'BadEnum')

    def testRepeatedInt(self):
        """Test duplicated numbers are forbidden."""
        self.assertRaises(messages.EnumDefinitionError, messages.Enum.def_enum, {'Ok': 1, 'Repeated': 1}, 'BadEnum')

    def testStr(self):
        """Test converting to string."""
        self.assertEquals('RED', str(Color.RED))
        self.assertEquals('ORANGE', str(Color.ORANGE))

    def testInt(self):
        """Test converting to int."""
        self.assertEquals(20, int(Color.RED))
        self.assertEquals(2, int(Color.ORANGE))

    def testRepr(self):
        """Test enum representation."""
        self.assertEquals('Color(RED, 20)', repr(Color.RED))
        self.assertEquals('Color(YELLOW, 40)', repr(Color.YELLOW))

    def testDocstring(self):
        """Test that docstring is supported ok."""

        class NotImportant(messages.Enum):
            """I have a docstring."""
            VALUE1 = 1
        self.assertEquals('I have a docstring.', NotImportant.__doc__)

    def testDeleteEnumValue(self):
        """Test that enum values cannot be deleted."""
        self.assertRaises(TypeError, delattr, Color, 'RED')

    def testEnumName(self):
        """Test enum name."""
        module_name = test_util.get_module_name(EnumTest)
        self.assertEquals('%s.Color' % module_name, Color.definition_name())
        self.assertEquals(module_name, Color.outer_definition_name())
        self.assertEquals(module_name, Color.definition_package())

    def testDefinitionName_OverrideModule(self):
        """Test enum module is overriden by module package name."""
        global package
        try:
            package = 'my.package'
            self.assertEquals('my.package.Color', Color.definition_name())
            self.assertEquals('my.package', Color.outer_definition_name())
            self.assertEquals('my.package', Color.definition_package())
        finally:
            del package

    def testDefinitionName_NoModule(self):
        """Test what happens when there is no module for enum."""

        class Enum1(messages.Enum):
            pass
        original_modules = sys.modules
        sys.modules = dict(sys.modules)
        try:
            del sys.modules[__name__]
            self.assertEquals('Enum1', Enum1.definition_name())
            self.assertEquals(None, Enum1.outer_definition_name())
            self.assertEquals(None, Enum1.definition_package())
            self.assertEquals(six.text_type, type(Enum1.definition_name()))
        finally:
            sys.modules = original_modules

    def testDefinitionName_Nested(self):
        """Test nested Enum names."""

        class MyMessage(messages.Message):

            class NestedEnum(messages.Enum):
                pass

            class NestedMessage(messages.Message):

                class NestedEnum(messages.Enum):
                    pass
        module_name = test_util.get_module_name(EnumTest)
        self.assertEquals('%s.MyMessage.NestedEnum' % module_name, MyMessage.NestedEnum.definition_name())
        self.assertEquals('%s.MyMessage' % module_name, MyMessage.NestedEnum.outer_definition_name())
        self.assertEquals(module_name, MyMessage.NestedEnum.definition_package())
        self.assertEquals('%s.MyMessage.NestedMessage.NestedEnum' % module_name, MyMessage.NestedMessage.NestedEnum.definition_name())
        self.assertEquals('%s.MyMessage.NestedMessage' % module_name, MyMessage.NestedMessage.NestedEnum.outer_definition_name())
        self.assertEquals(module_name, MyMessage.NestedMessage.NestedEnum.definition_package())

    def testMessageDefinition(self):
        """Test that enumeration knows its enclosing message definition."""

        class OuterEnum(messages.Enum):
            pass
        self.assertEquals(None, OuterEnum.message_definition())

        class OuterMessage(messages.Message):

            class InnerEnum(messages.Enum):
                pass
        self.assertEquals(OuterMessage, OuterMessage.InnerEnum.message_definition())

    def testComparison(self):
        """Test comparing various enums to different types."""

        class Enum1(messages.Enum):
            VAL1 = 1
            VAL2 = 2

        class Enum2(messages.Enum):
            VAL1 = 1
        self.assertEquals(Enum1.VAL1, Enum1.VAL1)
        self.assertNotEquals(Enum1.VAL1, Enum1.VAL2)
        self.assertNotEquals(Enum1.VAL1, Enum2.VAL1)
        self.assertNotEquals(Enum1.VAL1, 'VAL1')
        self.assertNotEquals(Enum1.VAL1, 1)
        self.assertNotEquals(Enum1.VAL1, 2)
        self.assertNotEquals(Enum1.VAL1, None)
        self.assertNotEquals(Enum1.VAL1, Enum2.VAL1)
        self.assertTrue(Enum1.VAL1 < Enum1.VAL2)
        self.assertTrue(Enum1.VAL2 > Enum1.VAL1)
        self.assertNotEquals(1, Enum2.VAL1)

    def testPickle(self):
        """Testing pickling and unpickling of Enum instances."""
        colors = list(Color)
        unpickled = pickle.loads(pickle.dumps(colors))
        self.assertEquals(colors, unpickled)
        for i, color in enumerate(colors):
            self.assertTrue(color is unpickled[i])