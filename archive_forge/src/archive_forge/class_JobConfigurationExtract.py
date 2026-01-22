from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobConfigurationExtract(_messages.Message):
    """A JobConfigurationExtract object.

  Fields:
    compression: [Optional] The compression type to use for exported files.
      Possible values include GZIP and NONE. The default value is NONE.
    destinationFormat: [Optional] The exported file format. Possible values
      include CSV, NEWLINE_DELIMITED_JSON and AVRO. The default value is CSV.
      Tables with nested or repeated fields cannot be exported as CSV.
    destinationUri: [Pick one] DEPRECATED: Use destinationUris instead,
      passing only one URI as necessary. The fully-qualified Google Cloud
      Storage URI where the extracted table should be written.
    destinationUris: [Pick one] A list of fully-qualified Google Cloud Storage
      URIs where the extracted table should be written.
    fieldDelimiter: [Optional] Delimiter to use between fields in the exported
      data. Default is ','
    printHeader: [Optional] Whether to print out a header row in the results.
      Default is true.
    sourceTable: [Required] A reference to the table being exported.
  """
    compression = _messages.StringField(1)
    destinationFormat = _messages.StringField(2)
    destinationUri = _messages.StringField(3)
    destinationUris = _messages.StringField(4, repeated=True)
    fieldDelimiter = _messages.StringField(5)
    printHeader = _messages.BooleanField(6, default=True)
    sourceTable = _messages.MessageField('TableReference', 7)