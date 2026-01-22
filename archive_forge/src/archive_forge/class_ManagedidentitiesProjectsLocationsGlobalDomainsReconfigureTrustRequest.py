from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsReconfigureTrustRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsReconfigureTrustRequest
  object.

  Fields:
    name: Required. The resource domain name, project name and location using
      the form: `projects/{project_id}/locations/global/domains/{domain_name}`
    reconfigureTrustRequest: A ReconfigureTrustRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    reconfigureTrustRequest = _messages.MessageField('ReconfigureTrustRequest', 2)