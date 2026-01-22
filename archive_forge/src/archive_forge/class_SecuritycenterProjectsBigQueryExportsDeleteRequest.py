from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterProjectsBigQueryExportsDeleteRequest(_messages.Message):
    """A SecuritycenterProjectsBigQueryExportsDeleteRequest object.

  Fields:
    name: Required. The name of the BigQuery export to delete. Its format is
      organizations/{organization}/bigQueryExports/{export_id},
      folders/{folder}/bigQueryExports/{export_id}, or
      projects/{project}/bigQueryExports/{export_id}
  """
    name = _messages.StringField(1, required=True)