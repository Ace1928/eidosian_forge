import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import util
class AnEnum(messages.Enum):
    value_one = 1
    value_two = 2