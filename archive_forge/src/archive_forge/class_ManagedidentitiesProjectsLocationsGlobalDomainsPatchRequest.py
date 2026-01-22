from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsPatchRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsPatchRequest object.

  Fields:
    domain: A Domain resource to be passed as the request body.
    name: Required. The unique name of the domain using the form:
      `projects/{project_id}/locations/global/domains/{domain_name}`.
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field. The elements of the repeated paths field may
      only include fields from Domain: * `labels` * `locations` *
      `authorized_networks` * `audit_logs_enabled`
  """
    domain = _messages.MessageField('Domain', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)