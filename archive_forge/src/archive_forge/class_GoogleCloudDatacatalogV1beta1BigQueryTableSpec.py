from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1BigQueryTableSpec(_messages.Message):
    """Describes a BigQuery table.

  Enums:
    TableSourceTypeValueValuesEnum: Output only. The table source type.

  Fields:
    tableSourceType: Output only. The table source type.
    tableSpec: Spec of a BigQuery table. This field should only be populated
      if `table_source_type` is `BIGQUERY_TABLE`.
    viewSpec: Table view specification. This field should only be populated if
      `table_source_type` is `BIGQUERY_VIEW`.
  """

    class TableSourceTypeValueValuesEnum(_messages.Enum):
        """Output only. The table source type.

    Values:
      TABLE_SOURCE_TYPE_UNSPECIFIED: Default unknown type.
      BIGQUERY_VIEW: Table view.
      BIGQUERY_TABLE: BigQuery native table.
      BIGQUERY_MATERIALIZED_VIEW: BigQuery materialized view.
    """
        TABLE_SOURCE_TYPE_UNSPECIFIED = 0
        BIGQUERY_VIEW = 1
        BIGQUERY_TABLE = 2
        BIGQUERY_MATERIALIZED_VIEW = 3
    tableSourceType = _messages.EnumField('TableSourceTypeValueValuesEnum', 1)
    tableSpec = _messages.MessageField('GoogleCloudDatacatalogV1beta1TableSpec', 2)
    viewSpec = _messages.MessageField('GoogleCloudDatacatalogV1beta1ViewSpec', 3)