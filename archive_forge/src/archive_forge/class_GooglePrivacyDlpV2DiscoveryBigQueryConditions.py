from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DiscoveryBigQueryConditions(_messages.Message):
    """Requirements that must be true before a table is scanned in discovery
  for the first time. There is an AND relationship between the top-level
  attributes. Additionally, minimum conditions with an OR relationship that
  must be met before Cloud DLP scans a table can be set (like a minimum row
  count or a minimum table age).

  Enums:
    TypeCollectionValueValuesEnum: Restrict discovery to categories of table
      types.

  Fields:
    createdAfter: BigQuery table must have been created after this date. Used
      to avoid backfilling.
    orConditions: At least one of the conditions must be true for a table to
      be scanned.
    typeCollection: Restrict discovery to categories of table types.
    types: Restrict discovery to specific table types.
  """

    class TypeCollectionValueValuesEnum(_messages.Enum):
        """Restrict discovery to categories of table types.

    Values:
      BIG_QUERY_COLLECTION_UNSPECIFIED: Unused.
      BIG_QUERY_COLLECTION_ALL_TYPES: Automatically generate profiles for all
        tables, even if the table type is not yet fully supported for
        analysis. Profiles for unsupported tables will be generated with
        errors to indicate their partial support. When full support is added,
        the tables will automatically be profiled during the next scheduled
        run.
      BIG_QUERY_COLLECTION_ONLY_SUPPORTED_TYPES: Only those types fully
        supported will be profiled. Will expand automatically as Cloud DLP
        adds support for new table types. Unsupported table types will not
        have partial profiles generated.
    """
        BIG_QUERY_COLLECTION_UNSPECIFIED = 0
        BIG_QUERY_COLLECTION_ALL_TYPES = 1
        BIG_QUERY_COLLECTION_ONLY_SUPPORTED_TYPES = 2
    createdAfter = _messages.StringField(1)
    orConditions = _messages.MessageField('GooglePrivacyDlpV2OrConditions', 2)
    typeCollection = _messages.EnumField('TypeCollectionValueValuesEnum', 3)
    types = _messages.MessageField('GooglePrivacyDlpV2BigQueryTableTypes', 4)