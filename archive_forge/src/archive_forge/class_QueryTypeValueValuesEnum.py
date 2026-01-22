from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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