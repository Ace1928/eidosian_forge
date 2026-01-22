from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3PageInfoFormInfo(_messages.Message):
    """Represents form information.

  Fields:
    parameterInfo: Optional for both WebhookRequest and WebhookResponse. The
      parameters contained in the form. Note that the webhook cannot add or
      remove any form parameter.
  """
    parameterInfo = _messages.MessageField('GoogleCloudDialogflowCxV3PageInfoFormInfoParameterInfo', 1, repeated=True)