from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1PageInfo(_messages.Message):
    """Represents page information communicated to and from the webhook.

  Fields:
    currentPage: Always present for WebhookRequest. Ignored for
      WebhookResponse. The unique identifier of the current page. Format:
      `projects//locations//agents//flows//pages/`.
    displayName: Always present for WebhookRequest. Ignored for
      WebhookResponse. The display name of the current page.
    formInfo: Optional for both WebhookRequest and WebhookResponse.
      Information about the form.
  """
    currentPage = _messages.StringField(1)
    displayName = _messages.StringField(2)
    formInfo = _messages.MessageField('GoogleCloudDialogflowCxV3beta1PageInfoFormInfo', 3)