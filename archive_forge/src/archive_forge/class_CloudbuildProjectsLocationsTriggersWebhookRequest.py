from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsTriggersWebhookRequest(_messages.Message):
    """A CloudbuildProjectsLocationsTriggersWebhookRequest object.

  Fields:
    httpBody: A HttpBody resource to be passed as the request body.
    name: The name of the `ReceiveTriggerWebhook` to retrieve. Format:
      `projects/{project}/locations/{location}/triggers/{trigger}`
    projectId: Project in which the specified trigger lives
    secret: Secret token used for authorization if an OAuth token isn't
      provided.
    trigger: Name of the trigger to run the payload against
  """
    httpBody = _messages.MessageField('HttpBody', 1)
    name = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3)
    secret = _messages.StringField(4)
    trigger = _messages.StringField(5)