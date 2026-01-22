import cgi
import datetime
import inspect
import os
import re
import socket
import types
import unittest
import six
from six.moves import range  # pylint: disable=redefined-builtin
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
class ProtoConformanceTestBase(object):
    """Protocol conformance test base class.

    Each supported protocol should implement two methods that support encoding
    and decoding of Message objects in that format:

      encode_message(message) - Serialize to encoding.
      encode_message(message, encoded_message) - Deserialize from encoding.

    Tests for the modules where these functions are implemented should extend
    this class in order to support basic behavioral expectations.  This ensures
    that protocols correctly encode and decode message transparently to the
    caller.

    In order to support these test, the base class should also extend
    the TestCase class and implement the following class attributes
    which define the encoded version of certain protocol buffers:

      encoded_partial:
        <OptionalMessage
          double_value: 1.23
          int64_value: -100000000000
          string_value: u"a string"
          enum_value: OptionalMessage.SimpleEnum.VAL2
          >

      encoded_full:
        <OptionalMessage
          double_value: 1.23
          float_value: -2.5
          int64_value: -100000000000
          uint64_value: 102020202020
          int32_value: 1020
          bool_value: true
          string_value: u"a stringя"
          bytes_value: b"a bytesÿþ"
          enum_value: OptionalMessage.SimpleEnum.VAL2
          >

      encoded_repeated:
        <RepeatedMessage
          double_value: [1.23, 2.3]
          float_value: [-2.5, 0.5]
          int64_value: [-100000000000, 20]
          uint64_value: [102020202020, 10]
          int32_value: [1020, 718]
          bool_value: [true, false]
          string_value: [u"a stringя", u"another string"]
          bytes_value: [b"a bytesÿþ", b"another bytes"]
          enum_value: [OptionalMessage.SimpleEnum.VAL2,
                       OptionalMessage.SimpleEnum.VAL 1]
          >

      encoded_nested:
        <HasNestedMessage
          nested: <NestedMessage
            a_value: "a string"
            >
          >

      encoded_repeated_nested:
        <HasNestedMessage
          repeated_nested: [
              <NestedMessage a_value: "a string">,
              <NestedMessage a_value: "another string">
            ]
          >

      unexpected_tag_message:
        An encoded message that has an undefined tag or number in the stream.

      encoded_default_assigned:
        <HasDefault
          a_value: "a default"
          >

      encoded_nested_empty:
        <HasOptionalNestedMessage
          nested: <OptionalMessage>
          >

      encoded_invalid_enum:
        <OptionalMessage
          enum_value: (invalid value for serialization type)
          >
    """
    encoded_empty_message = ''

    def testEncodeInvalidMessage(self):
        message = NestedMessage()
        self.assertRaises(messages.ValidationError, self.PROTOLIB.encode_message, message)

    def CompareEncoded(self, expected_encoded, actual_encoded):
        """Compare two encoded protocol values.

        Can be overridden by sub-classes to special case comparison.
        For example, to eliminate white space from output that is not
        relevant to encoding.

        Args:
          expected_encoded: Expected string encoded value.
          actual_encoded: Actual string encoded value.
        """
        self.assertEquals(expected_encoded, actual_encoded)

    def EncodeDecode(self, encoded, expected_message):
        message = self.PROTOLIB.decode_message(type(expected_message), encoded)
        self.assertEquals(expected_message, message)
        self.CompareEncoded(encoded, self.PROTOLIB.encode_message(message))

    def testEmptyMessage(self):
        self.EncodeDecode(self.encoded_empty_message, OptionalMessage())

    def testPartial(self):
        """Test message with a few values set."""
        message = OptionalMessage()
        message.double_value = 1.23
        message.int64_value = -100000000000
        message.int32_value = 1020
        message.string_value = u'a string'
        message.enum_value = OptionalMessage.SimpleEnum.VAL2
        self.EncodeDecode(self.encoded_partial, message)

    def testFull(self):
        """Test all types."""
        message = OptionalMessage()
        message.double_value = 1.23
        message.float_value = -2.5
        message.int64_value = -100000000000
        message.uint64_value = 102020202020
        message.int32_value = 1020
        message.bool_value = True
        message.string_value = u'a stringя'
        message.bytes_value = b'a bytes\xff\xfe'
        message.enum_value = OptionalMessage.SimpleEnum.VAL2
        self.EncodeDecode(self.encoded_full, message)

    def testRepeated(self):
        """Test repeated fields."""
        message = RepeatedMessage()
        message.double_value = [1.23, 2.3]
        message.float_value = [-2.5, 0.5]
        message.int64_value = [-100000000000, 20]
        message.uint64_value = [102020202020, 10]
        message.int32_value = [1020, 718]
        message.bool_value = [True, False]
        message.string_value = [u'a stringя', u'another string']
        message.bytes_value = [b'a bytes\xff\xfe', b'another bytes']
        message.enum_value = [RepeatedMessage.SimpleEnum.VAL2, RepeatedMessage.SimpleEnum.VAL1]
        self.EncodeDecode(self.encoded_repeated, message)

    def testNested(self):
        """Test nested messages."""
        nested_message = NestedMessage()
        nested_message.a_value = u'a string'
        message = HasNestedMessage()
        message.nested = nested_message
        self.EncodeDecode(self.encoded_nested, message)

    def testRepeatedNested(self):
        """Test repeated nested messages."""
        nested_message1 = NestedMessage()
        nested_message1.a_value = u'a string'
        nested_message2 = NestedMessage()
        nested_message2.a_value = u'another string'
        message = HasNestedMessage()
        message.repeated_nested = [nested_message1, nested_message2]
        self.EncodeDecode(self.encoded_repeated_nested, message)

    def testStringTypes(self):
        """Test that encoding str on StringField works."""
        message = OptionalMessage()
        message.string_value = 'Latin'
        self.EncodeDecode(self.encoded_string_types, message)

    def testEncodeUninitialized(self):
        """Test that cannot encode uninitialized message."""
        required = NestedMessage()
        self.assertRaisesWithRegexpMatch(messages.ValidationError, 'Message NestedMessage is missing required field a_value', self.PROTOLIB.encode_message, required)

    def testUnexpectedField(self):
        """Test decoding and encoding unexpected fields."""
        loaded_message = self.PROTOLIB.decode_message(OptionalMessage, self.unexpected_tag_message)
        self.assertEquals(OptionalMessage(), loaded_message)
        self.assertEquals(self.unexpected_tag_message, self.PROTOLIB.encode_message(loaded_message))

    def testDoNotSendDefault(self):
        """Test that default is not sent when nothing is assigned."""
        self.EncodeDecode(self.encoded_empty_message, HasDefault())

    def testSendDefaultExplicitlyAssigned(self):
        """Test that default is sent when explcitly assigned."""
        message = HasDefault()
        message.a_value = HasDefault.a_value.default
        self.EncodeDecode(self.encoded_default_assigned, message)

    def testEncodingNestedEmptyMessage(self):
        """Test encoding a nested empty message."""
        message = HasOptionalNestedMessage()
        message.nested = OptionalMessage()
        self.EncodeDecode(self.encoded_nested_empty, message)

    def testEncodingRepeatedNestedEmptyMessage(self):
        """Test encoding a nested empty message."""
        message = HasOptionalNestedMessage()
        message.repeated_nested = [OptionalMessage(), OptionalMessage()]
        self.EncodeDecode(self.encoded_repeated_nested_empty, message)

    def testContentType(self):
        self.assertTrue(isinstance(self.PROTOLIB.CONTENT_TYPE, str))

    def testDecodeInvalidEnumType(self):
        decoded = self.PROTOLIB.decode_message(OptionalMessage, self.encoded_invalid_enum)
        message = OptionalMessage()
        self.assertEqual(message, decoded)
        encoded = self.PROTOLIB.encode_message(decoded)
        self.assertEqual(self.encoded_invalid_enum, encoded)

    def testDateTimeNoTimeZone(self):
        """Test that DateTimeFields are encoded/decoded correctly."""

        class MyMessage(messages.Message):
            value = message_types.DateTimeField(1)
        value = datetime.datetime(2013, 1, 3, 11, 36, 30, 123000)
        message = MyMessage(value=value)
        decoded = self.PROTOLIB.decode_message(MyMessage, self.PROTOLIB.encode_message(message))
        self.assertEquals(decoded.value, value)

    def testDateTimeWithTimeZone(self):
        """Test DateTimeFields with time zones."""

        class MyMessage(messages.Message):
            value = message_types.DateTimeField(1)
        value = datetime.datetime(2013, 1, 3, 11, 36, 30, 123000, util.TimeZoneOffset(8 * 60))
        message = MyMessage(value=value)
        decoded = self.PROTOLIB.decode_message(MyMessage, self.PROTOLIB.encode_message(message))
        self.assertEquals(decoded.value, value)