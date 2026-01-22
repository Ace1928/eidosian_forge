from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AccessSet(_messages.Message):
    """Set action. For example, "Set" : { "name" : "target.name", "success" :
  true, "value" : "default" }

  Fields:
    name: A string attribute.
    success: A boolean attribute.
    value: A string attribute.
  """
    name = _messages.StringField(1)
    success = _messages.BooleanField(2)
    value = _messages.StringField(3)