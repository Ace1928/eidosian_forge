from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisualizationData(_messages.Message):
    """A VisualizationData object.

  Enums:
    KeyUnitValueValuesEnum: The unit for the key: e.g. 'key' or 'chunk'.

  Fields:
    dataSourceEndToken: The token signifying the end of a data_source.
    dataSourceSeparatorToken: The token delimiting a datasource name from the
      rest of a key in a data_source.
    diagnosticMessages: The list of messages (info, alerts, ...)
    endKeyStrings: We discretize the entire keyspace into buckets. Assuming
      each bucket has an inclusive keyrange and covers keys from k(i) ...
      k(n). In this case k(n) would be an end key for a given range.
      end_key_string is the collection of all such end keys
    hasPii: Whether this scan contains PII.
    indexedKeys: Keys of key ranges that contribute significantly to a given
      metric Can be thought of as heavy hitters.
    keySeparator: The token delimiting the key prefixes.
    keyUnit: The unit for the key: e.g. 'key' or 'chunk'.
    metrics: The list of data objects for each metric.
    prefixNodes: The list of extracted key prefix nodes used in the key prefix
      hierarchy.
  """

    class KeyUnitValueValuesEnum(_messages.Enum):
        """The unit for the key: e.g. 'key' or 'chunk'.

    Values:
      KEY_UNIT_UNSPECIFIED: Required default value
      KEY: Each entry corresponds to one key
      CHUNK: Each entry corresponds to a chunk of keys
    """
        KEY_UNIT_UNSPECIFIED = 0
        KEY = 1
        CHUNK = 2
    dataSourceEndToken = _messages.StringField(1)
    dataSourceSeparatorToken = _messages.StringField(2)
    diagnosticMessages = _messages.MessageField('DiagnosticMessage', 3, repeated=True)
    endKeyStrings = _messages.StringField(4, repeated=True)
    hasPii = _messages.BooleanField(5)
    indexedKeys = _messages.StringField(6, repeated=True)
    keySeparator = _messages.StringField(7)
    keyUnit = _messages.EnumField('KeyUnitValueValuesEnum', 8)
    metrics = _messages.MessageField('Metric', 9, repeated=True)
    prefixNodes = _messages.MessageField('PrefixNode', 10, repeated=True)