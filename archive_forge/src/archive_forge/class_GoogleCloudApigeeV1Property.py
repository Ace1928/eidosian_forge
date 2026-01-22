from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Property(_messages.Message):
    """A single property entry in the Properties message.

  Fields:
    name: The property key
    value: The property value
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)