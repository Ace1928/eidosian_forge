from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CustomResourceConversion(_messages.Message):
    """CustomResourceConversion describes how to convert different versions of
  a CR.

  Fields:
    strategy: strategy specifies how custom resources are converted between
      versions. Allowed values are: - `None`: The converter only change the
      apiVersion and would not touch any other field in the custom resource. -
      `Webhook`: API Server will call to an external webhook to do the
      conversion. Additional information is needed for this option. This
      requires spec.preserveUnknownFields to be false, and
      spec.conversion.webhook to be set.
    webhook: webhook describes how to call the conversion webhook. Required
      when `strategy` is set to `Webhook`.
  """
    strategy = _messages.StringField(1)
    webhook = _messages.MessageField('WebhookConversion', 2)