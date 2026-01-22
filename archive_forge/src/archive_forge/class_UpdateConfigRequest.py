from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateConfigRequest(_messages.Message):
    """Request message for the UpdateConfig method.

  Fields:
    config: Required. The config to update. The config's `name` field is used
      to identify the config to be updated. The expected format is
      `projects/{project}/locations/{location}/config`.
    updateMask: The list of fields to be updated.
  """
    config = _messages.MessageField('Config', 1)
    updateMask = _messages.StringField(2)