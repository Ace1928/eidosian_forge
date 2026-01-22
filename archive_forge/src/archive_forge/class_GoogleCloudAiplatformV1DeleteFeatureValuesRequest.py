from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DeleteFeatureValuesRequest(_messages.Message):
    """Request message for FeaturestoreService.DeleteFeatureValues.

  Fields:
    selectEntity: Select feature values to be deleted by specifying entities.
    selectTimeRangeAndFeature: Select feature values to be deleted by
      specifying time range and features.
  """
    selectEntity = _messages.MessageField('GoogleCloudAiplatformV1DeleteFeatureValuesRequestSelectEntity', 1)
    selectTimeRangeAndFeature = _messages.MessageField('GoogleCloudAiplatformV1DeleteFeatureValuesRequestSelectTimeRangeAndFeature', 2)