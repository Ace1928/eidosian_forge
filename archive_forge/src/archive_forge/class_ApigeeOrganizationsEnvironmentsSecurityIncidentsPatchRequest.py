from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityIncidentsPatchRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityIncidentsPatchRequest object.

  Fields:
    googleCloudApigeeV1SecurityIncident: A GoogleCloudApigeeV1SecurityIncident
      resource to be passed as the request body.
    name: Immutable. Name of the security incident resource. Format: organizat
      ions/{org}/environments/{environment}/securityIncidents/{incident}
      Example: organizations/apigee-
      org/environments/dev/securityIncidents/1234-5678-9101-1111
    updateMask: Required. The list of fields to update. Allowed fields are:
      LINT.IfChange(allowed_update_fields_comment) - observability
      LINT.ThenChange()
  """
    googleCloudApigeeV1SecurityIncident = _messages.MessageField('GoogleCloudApigeeV1SecurityIncident', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)