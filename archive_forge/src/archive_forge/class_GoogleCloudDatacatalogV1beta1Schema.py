from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1Schema(_messages.Message):
    """Represents a schema (e.g. BigQuery, GoogleSQL, Avro schema).

  Fields:
    columns: Required. Schema of columns. A maximum of 10,000 columns and sub-
      columns can be specified.
  """
    columns = _messages.MessageField('GoogleCloudDatacatalogV1beta1ColumnSchema', 1, repeated=True)