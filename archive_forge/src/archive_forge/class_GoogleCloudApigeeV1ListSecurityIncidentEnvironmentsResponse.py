from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListSecurityIncidentEnvironmentsResponse(_messages.Message):
    """Response for ListEnvironmentSecurityIncident.

  Fields:
    nextPageToken: Output only. A token that can be sent as `page_token` to
      retrieve the next page. If this field is omitted, there are no
      subsequent pages.
    securityIncidentEnvironments: List of environments with security incident
      stats.
  """
    nextPageToken = _messages.StringField(1)
    securityIncidentEnvironments = _messages.MessageField('GoogleCloudApigeeV1SecurityIncidentEnvironment', 2, repeated=True)