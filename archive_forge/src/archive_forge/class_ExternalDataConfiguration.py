from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExternalDataConfiguration(_messages.Message):
    """A ExternalDataConfiguration object.

  Fields:
    autodetect: [Experimental] Try to detect schema and format options
      automatically. Any option specified explicitly will be honored.
    bigtableOptions: [Optional] Additional options if sourceFormat is set to
      BIGTABLE.
    compression: [Optional] The compression type of the data source. Possible
      values include GZIP and NONE. The default value is NONE. This setting is
      ignored for Google Cloud Bigtable, Google Cloud Datastore backups and
      Avro formats.
    csvOptions: Additional properties to set if sourceFormat is set to CSV.
    googleSheetsOptions: [Optional] Additional options if sourceFormat is set
      to GOOGLE_SHEETS.
    ignoreUnknownValues: [Optional] Indicates if BigQuery should allow extra
      values that are not represented in the table schema. If true, the extra
      values are ignored. If false, records with extra columns are treated as
      bad records, and if there are too many bad records, an invalid error is
      returned in the job result. The default value is false. The sourceFormat
      property determines what BigQuery treats as an extra value: CSV:
      Trailing columns JSON: Named values that don't match any column names
      Google Cloud Bigtable: This setting is ignored. Google Cloud Datastore
      backups: This setting is ignored. Avro: This setting is ignored.
    maxBadRecords: [Optional] The maximum number of bad records that BigQuery
      can ignore when reading data. If the number of bad records exceeds this
      value, an invalid error is returned in the job result. The default value
      is 0, which requires that all records are valid. This setting is ignored
      for Google Cloud Bigtable, Google Cloud Datastore backups and Avro
      formats.
    schema: [Optional] The schema for the data. Schema is required for CSV and
      JSON formats. Schema is disallowed for Google Cloud Bigtable, Cloud
      Datastore backups, and Avro formats.
    sourceFormat: [Required] The data format. For CSV files, specify "CSV".
      For Google sheets, specify "GOOGLE_SHEETS". For newline-delimited JSON,
      specify "NEWLINE_DELIMITED_JSON". For Avro files, specify "AVRO". For
      Google Cloud Datastore backups, specify "DATASTORE_BACKUP".
      [Experimental] For Google Cloud Bigtable, specify "BIGTABLE". Please
      note that reading from Google Cloud Bigtable is experimental and has to
      be enabled for your project. Please contact Google Cloud Support to
      enable this for your project.
    sourceUris: [Required] The fully-qualified URIs that point to your data in
      Google Cloud. For Google Cloud Storage URIs: Each URI can contain one
      '*' wildcard character and it must come after the 'bucket' name. Size
      limits related to load jobs apply to external data sources. For Google
      Cloud Bigtable URIs: Exactly one URI can be specified and it has be a
      fully specified and valid HTTPS URL for a Google Cloud Bigtable table.
      For Google Cloud Datastore backups, exactly one URI can be specified,
      and it must end with '.backup_info'. Also, the '*' wildcard character is
      not allowed.
  """
    autodetect = _messages.BooleanField(1)
    bigtableOptions = _messages.MessageField('BigtableOptions', 2)
    compression = _messages.StringField(3)
    csvOptions = _messages.MessageField('CsvOptions', 4)
    googleSheetsOptions = _messages.MessageField('GoogleSheetsOptions', 5)
    ignoreUnknownValues = _messages.BooleanField(6)
    maxBadRecords = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    schema = _messages.MessageField('TableSchema', 8)
    sourceFormat = _messages.StringField(9)
    sourceUris = _messages.StringField(10, repeated=True)