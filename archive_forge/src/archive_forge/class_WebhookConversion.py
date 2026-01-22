from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WebhookConversion(_messages.Message):
    """WebhookConversion describes how to call a conversion webhook

  Fields:
    clientConfig: clientConfig is the instructions for how to call the webhook
      if strategy is `Webhook`.
    conversionReviewVersions: conversionReviewVersions is an ordered list of
      preferred `ConversionReview` versions the Webhook expects. The API
      server will use the first version in the list which it supports. If none
      of the versions specified in this list are supported by API server,
      conversion will fail for the custom resource. If a persisted Webhook
      configuration specifies allowed versions and does not include any
      versions known to the API Server, calls to the webhook will fail.
  """
    clientConfig = _messages.MessageField('WebhookClientConfig', 1)
    conversionReviewVersions = _messages.StringField(2, repeated=True)