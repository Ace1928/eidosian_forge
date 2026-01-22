from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NearestNeighborQueryParameters(_messages.Message):
    """Parameters that can be overrided in each query to tune query latency and
  recall.

  Fields:
    approximateNeighborCandidates: Optional. The number of neighbors to find
      via approximate search before exact reordering is performed; if set,
      this value must be > neighbor_count.
    leafNodesSearchFraction: Optional. The fraction of the number of leaves to
      search, set at query time allows user to tune search performance. This
      value increase result in both search accuracy and latency increase. The
      value should be between 0.0 and 1.0.
  """
    approximateNeighborCandidates = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    leafNodesSearchFraction = _messages.FloatField(2)