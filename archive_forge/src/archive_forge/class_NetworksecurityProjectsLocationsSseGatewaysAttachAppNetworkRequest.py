from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsSseGatewaysAttachAppNetworkRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsSseGatewaysAttachAppNetworkRequest
  object.

  Fields:
    attachAppNetworkRequest: A AttachAppNetworkRequest resource to be passed
      as the request body.
    name: Required. Name of the SSEGateway which will hold the attached
      network. Must be in the format
      `projects/*/locations/{location}/sseGateways/*`.
  """
    attachAppNetworkRequest = _messages.MessageField('AttachAppNetworkRequest', 1)
    name = _messages.StringField(2, required=True)