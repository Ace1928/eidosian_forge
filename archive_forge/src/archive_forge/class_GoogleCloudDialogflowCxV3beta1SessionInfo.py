from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1SessionInfo(_messages.Message):
    """Represents session information communicated to and from the webhook.

  Messages:
    ParametersValue: Optional for WebhookRequest. Optional for
      WebhookResponse. All parameters collected from forms and intents during
      the session. Parameters can be created, updated, or removed by the
      webhook. To remove a parameter from the session, the webhook should
      explicitly set the parameter value to null in WebhookResponse. The map
      is keyed by parameters' display names.

  Fields:
    parameters: Optional for WebhookRequest. Optional for WebhookResponse. All
      parameters collected from forms and intents during the session.
      Parameters can be created, updated, or removed by the webhook. To remove
      a parameter from the session, the webhook should explicitly set the
      parameter value to null in WebhookResponse. The map is keyed by
      parameters' display names.
    session: Always present for WebhookRequest. Ignored for WebhookResponse.
      The unique identifier of the session. This field can be used by the
      webhook to identify a session. Format:
      `projects//locations//agents//sessions/` or
      `projects//locations//agents//environments//sessions/` if environment is
      specified.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """Optional for WebhookRequest. Optional for WebhookResponse. All
    parameters collected from forms and intents during the session. Parameters
    can be created, updated, or removed by the webhook. To remove a parameter
    from the session, the webhook should explicitly set the parameter value to
    null in WebhookResponse. The map is keyed by parameters' display names.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Additional properties of type ParametersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    parameters = _messages.MessageField('ParametersValue', 1)
    session = _messages.StringField(2)