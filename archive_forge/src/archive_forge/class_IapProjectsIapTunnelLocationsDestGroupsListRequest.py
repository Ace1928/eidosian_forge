from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapProjectsIapTunnelLocationsDestGroupsListRequest(_messages.Message):
    """A IapProjectsIapTunnelLocationsDestGroupsListRequest object.

  Fields:
    pageSize: The maximum number of groups to return. The service might return
      fewer than this value. If unspecified, at most 100 groups are returned.
      The maximum value is 1000; values above 1000 are coerced to 1000.
    pageToken: A page token, received from a previous `ListTunnelDestGroups`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListTunnelDestGroups` must match the call
      that provided the page token.
    parent: Required. Google Cloud Project ID and location. In the following
      format: `projects/{project_number/id}/iap_tunnel/locations/{location}`.
      A `-` can be used for the location to group across all locations.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)