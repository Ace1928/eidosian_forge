import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class DescribeMessageTest(test_util.TestCase):

    def testEmptyDefinition(self):

        class MyMessage(messages.Message):
            pass
        expected = descriptor.MessageDescriptor()
        expected.name = 'MyMessage'
        described = descriptor.describe_message(MyMessage)
        described.check_initialized()
        self.assertEquals(expected, described)

    def testDefinitionWithFields(self):

        class MessageWithFields(messages.Message):
            field1 = messages.IntegerField(10)
            field2 = messages.StringField(30)
            field3 = messages.IntegerField(20)
        expected = descriptor.MessageDescriptor()
        expected.name = 'MessageWithFields'
        expected.fields = [descriptor.describe_field(MessageWithFields.field_by_name('field1')), descriptor.describe_field(MessageWithFields.field_by_name('field3')), descriptor.describe_field(MessageWithFields.field_by_name('field2'))]
        described = descriptor.describe_message(MessageWithFields)
        described.check_initialized()
        self.assertEquals(expected, described)

    def testNestedEnum(self):

        class MessageWithEnum(messages.Message):

            class Mood(messages.Enum):
                GOOD = 1
                BAD = 2
                UGLY = 3

            class Music(messages.Enum):
                CLASSIC = 1
                JAZZ = 2
                BLUES = 3
        expected = descriptor.MessageDescriptor()
        expected.name = 'MessageWithEnum'
        expected.enum_types = [descriptor.describe_enum(MessageWithEnum.Mood), descriptor.describe_enum(MessageWithEnum.Music)]
        described = descriptor.describe_message(MessageWithEnum)
        described.check_initialized()
        self.assertEquals(expected, described)

    def testNestedMessage(self):

        class MessageWithMessage(messages.Message):

            class Nesty(messages.Message):
                pass
        expected = descriptor.MessageDescriptor()
        expected.name = 'MessageWithMessage'
        expected.message_types = [descriptor.describe_message(MessageWithMessage.Nesty)]
        described = descriptor.describe_message(MessageWithMessage)
        described.check_initialized()
        self.assertEquals(expected, described)