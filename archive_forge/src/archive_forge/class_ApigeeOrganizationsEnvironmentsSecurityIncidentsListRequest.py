from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityIncidentsListRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityIncidentsListRequest object.

  Fields:
    filter: The filter expression to be used to get the list of security
      incidents, where filtering can be done on API Proxies. Example: filter =
      "api_proxy = /", "first_detected_time >", "last_detected_time <"
    pageSize: Optional. The maximum number of incidents to return. The service
      may return fewer than this value. If unspecified, at most 50 incidents
      will be returned.
    pageToken: Optional. A page token, received from a previous
      `ListSecurityIncident` call. Provide this to retrieve the subsequent
      page.
    parent: Required. For a specific organization, list of all the security
      incidents. Format: `organizations/{org}/environments/{environment}`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)