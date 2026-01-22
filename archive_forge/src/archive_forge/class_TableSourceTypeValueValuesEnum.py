from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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