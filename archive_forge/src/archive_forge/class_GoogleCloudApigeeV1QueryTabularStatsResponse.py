from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1QueryTabularStatsResponse(_messages.Message):
    """Encapsulates two kinds of stats that are results of the dimensions and
  aggregations requested. - Tabular rows. - Time series data. Example of
  tabular rows, Represents security stats results as a row of flat values.

  Messages:
    ValuesValueListEntry: Single entry in a ValuesValue.

  Fields:
    columns: Column names corresponding to the same order as the inner values
      in the stats field.
    nextPageToken: Next page token.
    values: Resultant rows from the executed query.
  """

    class ValuesValueListEntry(_messages.Message):
        """Single entry in a ValuesValue.

    Fields:
      entry: A extra_types.JsonValue attribute.
    """
        entry = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)
    columns = _messages.StringField(1, repeated=True)
    nextPageToken = _messages.StringField(2)
    values = _messages.MessageField('ValuesValueListEntry', 3, repeated=True)