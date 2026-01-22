from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityIncidentsBatchUpdateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityIncidentsBatchUpdateRequest
  object.

  Fields:
    googleCloudApigeeV1BatchUpdateSecurityIncidentsRequest: A
      GoogleCloudApigeeV1BatchUpdateSecurityIncidentsRequest resource to be
      passed as the request body.
    parent: Optional. The parent resource shared by all security incidents
      being updated. If this is set, the parent field in the
      UpdateSecurityIncidentRequest messages must either be empty or match
      this field.
  """
    googleCloudApigeeV1BatchUpdateSecurityIncidentsRequest = _messages.MessageField('GoogleCloudApigeeV1BatchUpdateSecurityIncidentsRequest', 1)
    parent = _messages.StringField(2, required=True)