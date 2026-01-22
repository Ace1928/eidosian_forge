import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
class MyMessage(messages.Message):
    """Test message containing various types."""

    class Color(messages.Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    class Nested(messages.Message):
        nested_value = messages.StringField(1)

    class NestedDatetime(messages.Message):
        nested_dt_value = message_types.DateTimeField(1)
    a_string = messages.StringField(2)
    an_integer = messages.IntegerField(3)
    a_float = messages.FloatField(4)
    a_boolean = messages.BooleanField(5)
    an_enum = messages.EnumField(Color, 6)
    a_nested = messages.MessageField(Nested, 7)
    a_repeated = messages.IntegerField(8, repeated=True)
    a_repeated_float = messages.FloatField(9, repeated=True)
    a_datetime = message_types.DateTimeField(10)
    a_repeated_datetime = message_types.DateTimeField(11, repeated=True)
    a_custom = CustomField(12)
    a_repeated_custom = CustomField(13, repeated=True)
    a_nested_datetime = messages.MessageField(NestedDatetime, 14)