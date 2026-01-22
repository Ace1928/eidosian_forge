from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FindNeighborsResponse(_messages.Message):
    """The response message for MatchService.FindNeighbors.

  Fields:
    nearestNeighbors: The nearest neighbors of the query datapoints.
  """
    nearestNeighbors = _messages.MessageField('GoogleCloudAiplatformV1beta1FindNeighborsResponseNearestNeighbors', 1, repeated=True)