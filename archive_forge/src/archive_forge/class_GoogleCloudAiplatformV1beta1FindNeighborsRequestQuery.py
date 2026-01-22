from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FindNeighborsRequestQuery(_messages.Message):
    """A query to find a number of the nearest neighbors (most similar vectors)
  of a vector.

  Fields:
    approximateNeighborCount: The number of neighbors to find via approximate
      search before exact reordering is performed. If not set, the default
      value from scam config is used; if set, this value must be > 0.
    datapoint: Required. The datapoint/vector whose nearest neighbors should
      be searched for.
    fractionLeafNodesToSearchOverride: The fraction of the number of leaves to
      search, set at query time allows user to tune search performance. This
      value increase result in both search accuracy and latency increase. The
      value should be between 0.0 and 1.0. If not set or set to 0.0, query
      uses the default value specified in
      NearestNeighborSearchConfig.TreeAHConfig.fraction_leaf_nodes_to_search.
    neighborCount: The number of nearest neighbors to be retrieved from
      database for each query. If not set, will use the default from the
      service configuration (https://cloud.google.com/vertex-ai/docs/matching-
      engine/configuring-indexes#nearest-neighbor-search-config).
    perCrowdingAttributeNeighborCount: Crowding is a constraint on a neighbor
      list produced by nearest neighbor search requiring that no more than
      some value k' of the k neighbors returned have the same value of
      crowding_attribute. It's used for improving result diversity. This field
      is the maximum number of matches with the same crowding tag.
  """
    approximateNeighborCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    datapoint = _messages.MessageField('GoogleCloudAiplatformV1beta1IndexDatapoint', 2)
    fractionLeafNodesToSearchOverride = _messages.FloatField(3)
    neighborCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    perCrowdingAttributeNeighborCount = _messages.IntegerField(5, variant=_messages.Variant.INT32)