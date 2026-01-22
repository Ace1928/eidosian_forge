from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ComputeTokensResponse(_messages.Message):
    """Response message for ComputeTokens RPC call.

  Fields:
    tokensInfo: Lists of tokens info from the input. A ComputeTokensRequest
      could have multiple instances with a prompt in each instance. We also
      need to return lists of tokens info for the request with multiple
      instances.
  """
    tokensInfo = _messages.MessageField('GoogleCloudAiplatformV1beta1TokensInfo', 1, repeated=True)