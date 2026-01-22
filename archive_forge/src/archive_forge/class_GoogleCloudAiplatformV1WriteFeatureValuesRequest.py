from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1WriteFeatureValuesRequest(_messages.Message):
    """Request message for FeaturestoreOnlineServingService.WriteFeatureValues.

  Fields:
    payloads: Required. The entities to be written. Up to 100,000 feature
      values can be written across all `payloads`.
  """
    payloads = _messages.MessageField('GoogleCloudAiplatformV1WriteFeatureValuesPayload', 1, repeated=True)