from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1DataplexTableSpec(_messages.Message):
    """Entry specification for a Dataplex table.

  Fields:
    dataplexSpec: Common Dataplex fields.
    externalTables: List of external tables registered by Dataplex in other
      systems based on the same underlying data. External tables allow to
      query this data in those systems.
    userManaged: Indicates if the table schema is managed by the user or not.
  """
    dataplexSpec = _messages.MessageField('GoogleCloudDatacatalogV1DataplexSpec', 1)
    externalTables = _messages.MessageField('GoogleCloudDatacatalogV1DataplexExternalTable', 2, repeated=True)
    userManaged = _messages.BooleanField(3)