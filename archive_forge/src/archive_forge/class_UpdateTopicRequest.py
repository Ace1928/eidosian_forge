from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class UpdateTopicRequest(_messages.Message):
    """Request for the UpdateTopic method.

  Fields:
    topic: Required. The updated topic object.
    updateMask: Required. Indicates which fields in the provided topic to
      update. Must be specified and non-empty. Note that if `update_mask`
      contains "message_storage_policy" but the `message_storage_policy` is
      not set in the `topic` provided above, then the updated value is
      determined by the policy configured at the project or organization
      level.
  """
    topic = _messages.MessageField('Topic', 1)
    updateMask = _messages.StringField(2)