from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsExtendSchemaRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsExtendSchemaRequest
  object.

  Fields:
    domain: Required. The domain resource name using the form:
      `projects/{project_id}/locations/global/domains/{domain_name}`
    extendSchemaRequest: A ExtendSchemaRequest resource to be passed as the
      request body.
  """
    domain = _messages.StringField(1, required=True)
    extendSchemaRequest = _messages.MessageField('ExtendSchemaRequest', 2)