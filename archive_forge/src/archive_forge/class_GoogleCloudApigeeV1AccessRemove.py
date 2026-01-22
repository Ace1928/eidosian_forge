from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AccessRemove(_messages.Message):
    """Remove action. For example, "Remove" : { "name" : "target.name",
  "success" : true }

  Fields:
    name: A string attribute.
    success: A boolean attribute.
  """
    name = _messages.StringField(1)
    success = _messages.BooleanField(2)