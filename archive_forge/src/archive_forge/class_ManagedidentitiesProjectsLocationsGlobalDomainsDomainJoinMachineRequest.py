from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsDomainJoinMachineRequest(_messages.Message):
    """A
  ManagedidentitiesProjectsLocationsGlobalDomainsDomainJoinMachineRequest
  object.

  Fields:
    domain: Required. The domain resource name using the form:
      projects/{project_id}/locations/global/domains/{domain_name}
    domainJoinMachineRequest: A DomainJoinMachineRequest resource to be passed
      as the request body.
  """
    domain = _messages.StringField(1, required=True)
    domainJoinMachineRequest = _messages.MessageField('DomainJoinMachineRequest', 2)