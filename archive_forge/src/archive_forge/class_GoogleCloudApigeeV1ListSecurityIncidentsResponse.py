from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListSecurityIncidentsResponse(_messages.Message):
    """Response for ListSecurityIncidents.

  Fields:
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    securityIncidents: List of security incidents in the organization
  """
    nextPageToken = _messages.StringField(1)
    securityIncidents = _messages.MessageField('GoogleCloudApigeeV1SecurityIncident', 2, repeated=True)