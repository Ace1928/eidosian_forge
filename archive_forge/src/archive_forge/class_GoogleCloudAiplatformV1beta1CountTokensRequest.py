from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1CountTokensRequest(_messages.Message):
    """Request message for PredictionService.CountTokens.

  Fields:
    contents: Required. Input content.
    instances: Required. The instances that are the input to token counting
      call. Schema is identical to the prediction schema of the underlying
      model.
    model: Required. The name of the publisher model requested to serve the
      prediction. Format:
      `projects/{project}/locations/{location}/publishers/*/models/*`
  """
    contents = _messages.MessageField('GoogleCloudAiplatformV1beta1Content', 1, repeated=True)
    instances = _messages.MessageField('extra_types.JsonValue', 2, repeated=True)
    model = _messages.StringField(3)