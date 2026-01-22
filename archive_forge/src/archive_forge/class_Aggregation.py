from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Aggregation(_messages.Message):
    """Defines an aggregation that produces a single result.

  Fields:
    alias: Optional. Optional name of the field to store the result of the
      aggregation into. If not provided, Firestore will pick a default name
      following the format `field_`. For example: ``` AGGREGATE COUNT_UP_TO(1)
      AS count_up_to_1, COUNT_UP_TO(2), COUNT_UP_TO(3) AS count_up_to_3,
      COUNT(*) OVER ( ... ); ``` becomes: ``` AGGREGATE COUNT_UP_TO(1) AS
      count_up_to_1, COUNT_UP_TO(2) AS field_1, COUNT_UP_TO(3) AS
      count_up_to_3, COUNT(*) AS field_2 OVER ( ... ); ``` Requires: * Must be
      unique across all aggregation aliases. * Conform to document field name
      limitations.
    avg: Average aggregator.
    count: Count aggregator.
    sum: Sum aggregator.
  """
    alias = _messages.StringField(1)
    avg = _messages.MessageField('Avg', 2)
    count = _messages.MessageField('Count', 3)
    sum = _messages.MessageField('Sum', 4)