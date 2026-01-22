from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AccessGet(_messages.Message):
    """Get action. For example, "Get" : { "name" : "target.name", "value" :
  "default" }

  Fields:
    name: A string attribute.
    value: A string attribute.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)