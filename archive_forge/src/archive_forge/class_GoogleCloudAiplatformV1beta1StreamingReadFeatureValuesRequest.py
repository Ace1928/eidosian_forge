from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StreamingReadFeatureValuesRequest(_messages.Message):
    """Request message for
  FeaturestoreOnlineServingService.StreamingFeatureValuesRead.

  Fields:
    entityIds: Required. IDs of entities to read Feature values of. The
      maximum number of IDs is 100. For example, for a machine learning model
      predicting user clicks on a website, an entity ID could be `user_123`.
    featureSelector: Required. Selector choosing Features of the target
      EntityType. Feature IDs will be deduplicated.
  """
    entityIds = _messages.StringField(1, repeated=True)
    featureSelector = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureSelector', 2)