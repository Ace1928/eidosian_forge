from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsSchemaExtensionsGetRequest(_messages.Message):
    """A
  ManagedidentitiesProjectsLocationsGlobalDomainsSchemaExtensionsGetRequest
  object.

  Fields:
    name: Required. Managed AD Schema Extension resource name using the form:
      `projects/{project_id}/locations/global/domains/{domain_name}/schemaExte
      nsions/{schema_extension_id}`
  """
    name = _messages.StringField(1, required=True)