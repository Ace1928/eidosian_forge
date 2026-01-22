from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SearchNearestEntitiesResponse(_messages.Message):
    """Response message for FeatureOnlineStoreService.SearchNearestEntities

  Fields:
    nearestNeighbors: The nearest neighbors of the query entity.
  """
    nearestNeighbors = _messages.MessageField('GoogleCloudAiplatformV1NearestNeighbors', 1)