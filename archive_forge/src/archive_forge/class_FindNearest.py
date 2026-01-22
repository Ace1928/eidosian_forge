from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FindNearest(_messages.Message):
    """Nearest Neighbors search config.

  Enums:
    DistanceMeasureValueValuesEnum: Required. The Distance Measure to use,
      required.

  Fields:
    distanceMeasure: Required. The Distance Measure to use, required.
    limit: Required. The number of nearest neighbors to return. Must be a
      positive integer of no more than 1000.
    queryVector: Required. The query vector that we are searching on. Must be
      a vector of no more than 2048 dimensions.
    vectorField: Required. An indexed vector field to search upon. Only
      documents which contain vectors whose dimensionality match the
      query_vector can be returned.
  """

    class DistanceMeasureValueValuesEnum(_messages.Enum):
        """Required. The Distance Measure to use, required.

    Values:
      DISTANCE_MEASURE_UNSPECIFIED: Should not be set.
      EUCLIDEAN: Measures the EUCLIDEAN distance between the vectors. See
        [Euclidean](https://en.wikipedia.org/wiki/Euclidean_distance) to learn
        more
      COSINE: Compares vectors based on the angle between them, which allows
        you to measure similarity that isn't based on the vectors magnitude.
        We recommend using DOT_PRODUCT with unit normalized vectors instead of
        COSINE distance, which is mathematically equivalent with better
        performance. See [Cosine
        Similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to learn
        more.
      DOT_PRODUCT: Similar to cosine but is affected by the magnitude of the
        vectors. See [Dot Product](https://en.wikipedia.org/wiki/Dot_product)
        to learn more.
    """
        DISTANCE_MEASURE_UNSPECIFIED = 0
        EUCLIDEAN = 1
        COSINE = 2
        DOT_PRODUCT = 3
    distanceMeasure = _messages.EnumField('DistanceMeasureValueValuesEnum', 1)
    limit = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    queryVector = _messages.MessageField('Value', 3)
    vectorField = _messages.MessageField('FieldReference', 4)