from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1IndexDatapoint(_messages.Message):
    """A datapoint of Index.

  Fields:
    crowdingTag: Optional. CrowdingTag of the datapoint, the number of
      neighbors to return in each crowding can be configured during query.
    datapointId: Required. Unique identifier of the datapoint.
    featureVector: Required. Feature embedding vector. An array of numbers
      with the length of [NearestNeighborSearchConfig.dimensions].
    numericRestricts: Optional. List of Restrict of the datapoint, used to
      perform "restricted searches" where boolean rule are used to filter the
      subset of the database eligible for matching. This uses numeric
      comparisons.
    restricts: Optional. List of Restrict of the datapoint, used to perform
      "restricted searches" where boolean rule are used to filter the subset
      of the database eligible for matching. This uses categorical tokens.
      See: https://cloud.google.com/vertex-ai/docs/matching-engine/filtering
  """
    crowdingTag = _messages.MessageField('GoogleCloudAiplatformV1beta1IndexDatapointCrowdingTag', 1)
    datapointId = _messages.StringField(2)
    featureVector = _messages.FloatField(3, repeated=True, variant=_messages.Variant.FLOAT)
    numericRestricts = _messages.MessageField('GoogleCloudAiplatformV1beta1IndexDatapointNumericRestriction', 4, repeated=True)
    restricts = _messages.MessageField('GoogleCloudAiplatformV1beta1IndexDatapointRestriction', 5, repeated=True)