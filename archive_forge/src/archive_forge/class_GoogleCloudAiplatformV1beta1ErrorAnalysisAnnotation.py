from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ErrorAnalysisAnnotation(_messages.Message):
    """Model error analysis for each annotation.

  Enums:
    QueryTypeValueValuesEnum: The query type used for finding the attributed
      items.

  Fields:
    attributedItems: Attributed items for a given annotation, typically
      representing neighbors from the training sets constrained by the query
      type.
    outlierScore: The outlier score of this annotated item. Usually defined as
      the min of all distances from attributed items.
    outlierThreshold: The threshold used to determine if this annotation is an
      outlier or not.
    queryType: The query type used for finding the attributed items.
  """

    class QueryTypeValueValuesEnum(_messages.Enum):
        """The query type used for finding the attributed items.

    Values:
      QUERY_TYPE_UNSPECIFIED: Unspecified query type for model error analysis.
      ALL_SIMILAR: Query similar samples across all classes in the dataset.
      SAME_CLASS_SIMILAR: Query similar samples from the same class of the
        input sample.
      SAME_CLASS_DISSIMILAR: Query dissimilar samples from the same class of
        the input sample.
    """
        QUERY_TYPE_UNSPECIFIED = 0
        ALL_SIMILAR = 1
        SAME_CLASS_SIMILAR = 2
        SAME_CLASS_DISSIMILAR = 3
    attributedItems = _messages.MessageField('GoogleCloudAiplatformV1beta1ErrorAnalysisAnnotationAttributedItem', 1, repeated=True)
    outlierScore = _messages.FloatField(2)
    outlierThreshold = _messages.FloatField(3)
    queryType = _messages.EnumField('QueryTypeValueValuesEnum', 4)