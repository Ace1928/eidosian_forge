from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3WebhookResponseFulfillmentResponse(_messages.Message):
    """Represents a fulfillment response to the user.

  Enums:
    MergeBehaviorValueValuesEnum: Merge behavior for `messages`.

  Fields:
    mergeBehavior: Merge behavior for `messages`.
    messages: The list of rich message responses to present to the user.
  """

    class MergeBehaviorValueValuesEnum(_messages.Enum):
        """Merge behavior for `messages`.

    Values:
      MERGE_BEHAVIOR_UNSPECIFIED: Not specified. `APPEND` will be used.
      APPEND: `messages` will be appended to the list of messages waiting to
        be sent to the user.
      REPLACE: `messages` will replace the list of messages waiting to be sent
        to the user.
    """
        MERGE_BEHAVIOR_UNSPECIFIED = 0
        APPEND = 1
        REPLACE = 2
    mergeBehavior = _messages.EnumField('MergeBehaviorValueValuesEnum', 1)
    messages = _messages.MessageField('GoogleCloudDialogflowCxV3ResponseMessage', 2, repeated=True)