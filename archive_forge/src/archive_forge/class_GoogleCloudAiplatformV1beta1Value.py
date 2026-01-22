from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Value(_messages.Message):
    """Value is the value of the field.

  Fields:
    doubleValue: A double value.
    intValue: An integer value.
    stringValue: A string value.
  """
    doubleValue = _messages.FloatField(1)
    intValue = _messages.IntegerField(2)
    stringValue = _messages.StringField(3)