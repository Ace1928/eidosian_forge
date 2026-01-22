import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class DescribeEnumTest(test_util.TestCase):

    def testEmptyEnum(self):

        class EmptyEnum(messages.Enum):
            pass
        expected = descriptor.EnumDescriptor()
        expected.name = 'EmptyEnum'
        described = descriptor.describe_enum(EmptyEnum)
        described.check_initialized()
        self.assertEquals(expected, described)

    def testNestedEnum(self):

        class MyScope(messages.Message):

            class NestedEnum(messages.Enum):
                pass
        expected = descriptor.EnumDescriptor()
        expected.name = 'NestedEnum'
        described = descriptor.describe_enum(MyScope.NestedEnum)
        described.check_initialized()
        self.assertEquals(expected, described)

    @unittest.skipIf('PyPy' in platform.python_implementation(), 'todo: reenable this')
    def testEnumWithItems(self):

        class EnumWithItems(messages.Enum):
            A = 3
            B = 1
            C = 2
        expected = descriptor.EnumDescriptor()
        expected.name = 'EnumWithItems'
        a = descriptor.EnumValueDescriptor()
        a.name = 'A'
        a.number = 3
        b = descriptor.EnumValueDescriptor()
        b.name = 'B'
        b.number = 1
        c = descriptor.EnumValueDescriptor()
        c.name = 'C'
        c.number = 2
        expected.values = [b, c, a]
        described = descriptor.describe_enum(EnumWithItems)
        described.check_initialized()
        self.assertEquals(expected, described)