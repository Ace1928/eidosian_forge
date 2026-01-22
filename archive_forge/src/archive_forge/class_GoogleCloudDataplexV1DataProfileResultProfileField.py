from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataProfileResultProfileField(_messages.Message):
    """A field within a table.

  Fields:
    mode: The mode of the field. Possible values include: REQUIRED, if it is a
      required field. NULLABLE, if it is an optional field. REPEATED, if it is
      a repeated field.
    name: The name of the field.
    profile: Profile information for the corresponding field.
    type: The data type retrieved from the schema of the data source. For
      instance, for a BigQuery native table, it is the BigQuery Table Schema (
      https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#tablefie
      ldschema). For a Dataplex Entity, it is the Entity Schema (https://cloud
      .google.com/dataplex/docs/reference/rpc/google.cloud.dataplex.v1#type_3)
      .
  """
    mode = _messages.StringField(1)
    name = _messages.StringField(2)
    profile = _messages.MessageField('GoogleCloudDataplexV1DataProfileResultProfileFieldProfileInfo', 3)
    type = _messages.StringField(4)