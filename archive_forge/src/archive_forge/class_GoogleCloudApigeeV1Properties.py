from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Properties(_messages.Message):
    """Message for compatibility with legacy Edge specification for Java
  Properties object in JSON.

  Fields:
    property: List of all properties in the object
  """
    property = _messages.MessageField('GoogleCloudApigeeV1Property', 1, repeated=True)