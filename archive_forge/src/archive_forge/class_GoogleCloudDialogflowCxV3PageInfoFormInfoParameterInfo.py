from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3PageInfoFormInfoParameterInfo(_messages.Message):
    """Represents parameter information.

  Enums:
    StateValueValuesEnum: Always present for WebhookRequest. Required for
      WebhookResponse. The state of the parameter. This field can be set to
      INVALID by the webhook to invalidate the parameter; other values set by
      the webhook will be ignored.

  Fields:
    displayName: Always present for WebhookRequest. Required for
      WebhookResponse. The human-readable name of the parameter, unique within
      the form. This field cannot be modified by the webhook.
    justCollected: Optional for WebhookRequest. Ignored for WebhookResponse.
      Indicates if the parameter value was just collected on the last
      conversation turn.
    required: Optional for both WebhookRequest and WebhookResponse. Indicates
      whether the parameter is required. Optional parameters will not trigger
      prompts; however, they are filled if the user specifies them. Required
      parameters must be filled before form filling concludes.
    state: Always present for WebhookRequest. Required for WebhookResponse.
      The state of the parameter. This field can be set to INVALID by the
      webhook to invalidate the parameter; other values set by the webhook
      will be ignored.
    value: Optional for both WebhookRequest and WebhookResponse. The value of
      the parameter. This field can be set by the webhook to change the
      parameter value.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Always present for WebhookRequest. Required for WebhookResponse. The
    state of the parameter. This field can be set to INVALID by the webhook to
    invalidate the parameter; other values set by the webhook will be ignored.

    Values:
      PARAMETER_STATE_UNSPECIFIED: Not specified. This value should be never
        used.
      EMPTY: Indicates that the parameter does not have a value.
      INVALID: Indicates that the parameter value is invalid. This field can
        be used by the webhook to invalidate the parameter and ask the server
        to collect it from the user again.
      FILLED: Indicates that the parameter has a value.
    """
        PARAMETER_STATE_UNSPECIFIED = 0
        EMPTY = 1
        INVALID = 2
        FILLED = 3
    displayName = _messages.StringField(1)
    justCollected = _messages.BooleanField(2)
    required = _messages.BooleanField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    value = _messages.MessageField('extra_types.JsonValue', 5)