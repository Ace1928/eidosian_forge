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
class BadMessage(messages.Message):
    f1 = messages.IntegerField(1)
    f2 = messages.IntegerField(1)