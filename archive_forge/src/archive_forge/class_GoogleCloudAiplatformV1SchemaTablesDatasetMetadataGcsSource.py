from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTablesDatasetMetadataGcsSource(_messages.Message):
    """A GoogleCloudAiplatformV1SchemaTablesDatasetMetadataGcsSource object.

  Fields:
    uri: Cloud Storage URI of one or more files. Only CSV files are supported.
      The first line of the CSV file is used as the header. If there are
      multiple files, the header is the first line of the lexicographically
      first file, the other files must either contain the exact same header or
      omit the header.
  """
    uri = _messages.StringField(1, repeated=True)