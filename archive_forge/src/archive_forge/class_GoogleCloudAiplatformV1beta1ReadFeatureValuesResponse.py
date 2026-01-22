from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ReadFeatureValuesResponse(_messages.Message):
    """Response message for FeaturestoreOnlineServingService.ReadFeatureValues.

  Fields:
    entityView: Entity view with Feature values. This may be the entity in the
      Featurestore if values for all Features were requested, or a projection
      of the entity in the Featurestore if values for only some Features were
      requested.
    header: Response header.
  """
    entityView = _messages.MessageField('GoogleCloudAiplatformV1beta1ReadFeatureValuesResponseEntityView', 1)
    header = _messages.MessageField('GoogleCloudAiplatformV1beta1ReadFeatureValuesResponseHeader', 2)