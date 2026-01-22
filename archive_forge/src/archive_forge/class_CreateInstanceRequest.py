from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateInstanceRequest(_messages.Message):
    """The request for CreateInstance.

  Fields:
    instance: Required. The instance to create. The name may be omitted, but
      if specified must be `/instances/`.
    instanceId: Required. The ID of the instance to create. Valid identifiers
      are of the form `a-z*[a-z0-9]` and must be between 2 and 64 characters
      in length.
  """
    instance = _messages.MessageField('Instance', 1)
    instanceId = _messages.StringField(2)