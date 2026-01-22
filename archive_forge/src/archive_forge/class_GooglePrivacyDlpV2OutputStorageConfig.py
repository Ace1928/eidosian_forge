from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2OutputStorageConfig(_messages.Message):
    """Cloud repository for storing output.

  Enums:
    OutputSchemaValueValuesEnum: Schema used for writing the findings for
      Inspect jobs. This field is only used for Inspect and must be
      unspecified for Risk jobs. Columns are derived from the `Finding`
      object. If appending to an existing table, any columns from the
      predefined schema that are missing will be added. No columns in the
      existing table will be deleted. If unspecified, then all available
      columns will be used for a new table or an (existing) table with no
      schema, and no changes will be made to an existing table that has a
      schema. Only for use with external storage.

  Fields:
    outputSchema: Schema used for writing the findings for Inspect jobs. This
      field is only used for Inspect and must be unspecified for Risk jobs.
      Columns are derived from the `Finding` object. If appending to an
      existing table, any columns from the predefined schema that are missing
      will be added. No columns in the existing table will be deleted. If
      unspecified, then all available columns will be used for a new table or
      an (existing) table with no schema, and no changes will be made to an
      existing table that has a schema. Only for use with external storage.
    table: Store findings in an existing table or a new table in an existing
      dataset. If table_id is not set a new one will be generated for you with
      the following format: dlp_googleapis_yyyy_mm_dd_[dlp_job_id]. Pacific
      time zone will be used for generating the date details. For Inspect,
      each column in an existing output table must have the same name, type,
      and mode of a field in the `Finding` object. For Risk, an existing
      output table should be the output of a previous Risk analysis job run on
      the same source table, with the same privacy metric and quasi-
      identifiers. Risk jobs that analyze the same table but compute a
      different privacy metric, or use different sets of quasi-identifiers,
      cannot store their results in the same table.
  """

    class OutputSchemaValueValuesEnum(_messages.Enum):
        """Schema used for writing the findings for Inspect jobs. This field is
    only used for Inspect and must be unspecified for Risk jobs. Columns are
    derived from the `Finding` object. If appending to an existing table, any
    columns from the predefined schema that are missing will be added. No
    columns in the existing table will be deleted. If unspecified, then all
    available columns will be used for a new table or an (existing) table with
    no schema, and no changes will be made to an existing table that has a
    schema. Only for use with external storage.

    Values:
      OUTPUT_SCHEMA_UNSPECIFIED: Unused.
      BASIC_COLUMNS: Basic schema including only `info_type`, `quote`,
        `certainty`, and `timestamp`.
      GCS_COLUMNS: Schema tailored to findings from scanning Cloud Storage.
      DATASTORE_COLUMNS: Schema tailored to findings from scanning Google
        Datastore.
      BIG_QUERY_COLUMNS: Schema tailored to findings from scanning Google
        BigQuery.
      ALL_COLUMNS: Schema containing all columns.
    """
        OUTPUT_SCHEMA_UNSPECIFIED = 0
        BASIC_COLUMNS = 1
        GCS_COLUMNS = 2
        DATASTORE_COLUMNS = 3
        BIG_QUERY_COLUMNS = 4
        ALL_COLUMNS = 5
    outputSchema = _messages.EnumField('OutputSchemaValueValuesEnum', 1)
    table = _messages.MessageField('GooglePrivacyDlpV2BigQueryTable', 2)