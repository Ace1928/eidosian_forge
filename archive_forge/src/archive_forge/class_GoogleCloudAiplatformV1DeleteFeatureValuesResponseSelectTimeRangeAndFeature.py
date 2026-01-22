from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DeleteFeatureValuesResponseSelectTimeRangeAndFeature(_messages.Message):
    """Response message if the request uses the SelectTimeRangeAndFeature
  option.

  Fields:
    impactedFeatureCount: The count of the features or columns impacted. This
      is the same as the feature count in the request.
    offlineStorageModifiedEntityRowCount: The count of modified entity rows in
      the offline storage. Each row corresponds to the combination of an
      entity ID and a timestamp. One entity ID can have multiple rows in the
      offline storage. Within each row, only the features specified in the
      request are deleted.
    onlineStorageModifiedEntityCount: The count of modified entities in the
      online storage. Each entity ID corresponds to one entity. Within each
      entity, only the features specified in the request are deleted.
  """
    impactedFeatureCount = _messages.IntegerField(1)
    offlineStorageModifiedEntityRowCount = _messages.IntegerField(2)
    onlineStorageModifiedEntityCount = _messages.IntegerField(3)