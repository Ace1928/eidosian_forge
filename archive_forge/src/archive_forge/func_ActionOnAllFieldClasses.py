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
def ActionOnAllFieldClasses(self, action):
    """Test all field classes except Message and Enum.

        Message and Enum require separate tests.

        Args:
          action: Callable that takes the field class as a parameter.
        """
    classes = (messages.IntegerField, messages.FloatField, messages.BooleanField, messages.BytesField, messages.StringField)
    for field_class in classes:
        action(field_class)