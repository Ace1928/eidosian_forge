from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancesSetNameRequest(_messages.Message):
    """A InstancesSetNameRequest object.

  Fields:
    currentName: The current name of this resource, used to prevent conflicts.
      Provide the latest name when making a request to change name.
    name: The name to be applied to the instance. Needs to be RFC 1035
      compliant.
  """
    currentName = _messages.StringField(1)
    name = _messages.StringField(2)