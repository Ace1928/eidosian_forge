from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsLocationsBigQueryExportsCreateRequest(_messages.Message):
    """A SecuritycenterOrganizationsLocationsBigQueryExportsCreateRequest
  object.

  Fields:
    bigQueryExportId: Required. Unique identifier provided by the client
      within the parent scope. It must consist of only lowercase letters,
      numbers, and hyphens, must start with a letter, must end with either a
      letter or a number, and must be 63 characters or less.
    googleCloudSecuritycenterV2BigQueryExport: A
      GoogleCloudSecuritycenterV2BigQueryExport resource to be passed as the
      request body.
    parent: Required. The name of the parent resource of the new BigQuery
      export. Its format is
      "organizations/[organization_id]/locations/[location_id]",
      "folders/[folder_id]/locations/[location_id]", or
      "projects/[project_id]/locations/[location_id]".
  """
    bigQueryExportId = _messages.StringField(1)
    googleCloudSecuritycenterV2BigQueryExport = _messages.MessageField('GoogleCloudSecuritycenterV2BigQueryExport', 2)
    parent = _messages.StringField(3, required=True)