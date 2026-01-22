from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryRestrictionConflict(_messages.Message):
    """A conflict within a query that prevents applying restrictions. For
  instance, if the query contains a timestamp, this conflicts with timestamp
  restrictions e.g. time picker settings.

  Enums:
    ConfidenceValueValuesEnum: How confident the detector is that the
      restriction would cause a conflict.
    TypeValueValuesEnum: Specifies what conflict is present. Currently, this
      only supports timerange.

  Fields:
    column: One-based column number where the conflict was detected within the
      query.
    confidence: How confident the detector is that the restriction would cause
      a conflict.
    line: One-based line number where the conflict was detected within the
      query.
    type: Specifies what conflict is present. Currently, this only supports
      timerange.
  """

    class ConfidenceValueValuesEnum(_messages.Enum):
        """How confident the detector is that the restriction would cause a
    conflict.

    Values:
      CONFIDENCE_UNSPECIFIED: Invalid.
      CERTAIN: If set, the query would be adversely affected by applying the
        restriction.
      UNCERTAIN: If set, the Query used a column being restricted, but might
        not be adversely affected.
    """
        CONFIDENCE_UNSPECIFIED = 0
        CERTAIN = 1
        UNCERTAIN = 2

    class TypeValueValuesEnum(_messages.Enum):
        """Specifies what conflict is present. Currently, this only supports
    timerange.

    Values:
      RESTRICTION_TYPE_UNSPECIFIED: Invalid.
      TIME_RANGE: This type means that the query conflicts with the time range
        restriction, e.g. query used the timestamp column to filter.
      JOIN: This type means that the query conflicts with a join restriction,
        meaning the query is using the JOIN operator.JOIN is important to
        detect for Ops Analytics Alerting queries because we want to prevent
        users from using potentially expensive JOIN based queries.
      LIMIT: This type means that the query conflicts with a limit
        restriction, meaning the query is using the LIMIT clause.LIMIT
        detection is going to be used for Ops Analytics Alerting hints towards
        the user to steer them away from including LIMIT in their queries.
    """
        RESTRICTION_TYPE_UNSPECIFIED = 0
        TIME_RANGE = 1
        JOIN = 2
        LIMIT = 3
    column = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    confidence = _messages.EnumField('ConfidenceValueValuesEnum', 2)
    line = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    type = _messages.EnumField('TypeValueValuesEnum', 4)