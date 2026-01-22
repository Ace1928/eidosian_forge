from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricsTypeValueValuesEnum(_messages.Enum):
    """The metrics type for the label.

    Values:
      METRICS_TYPE_UNSPECIFIED: The metrics type is unspecified. By default,
        metrics without a particular specification are for leaf entity types
        (i.e., top-level entity types without child types, or child types
        which are not parent types themselves).
      AGGREGATE: Indicates whether metrics for this particular label type
        represent an aggregate of metrics for other types instead of being
        based on actual TP/FP/FN values for the label type. Metrics for parent
        (i.e., non-leaf) entity types are an aggregate of metrics for their
        children.
    """
    METRICS_TYPE_UNSPECIFIED = 0
    AGGREGATE = 1