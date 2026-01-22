from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityOrganizationsLocationsSecurityProfilesListRequest(_messages.Message):
    """A NetworksecurityOrganizationsLocationsSecurityProfilesListRequest
  object.

  Fields:
    pageSize: Maximum number of SecurityProfiles to return per call.
    pageToken: The value returned by the last `ListSecurityProfilesResponse`
      Indicates that this is a continuation of a prior `ListSecurityProfiles`
      call, and that the system should return the next page of data.
    parent: Required. The project or organization and location from which the
      SecurityProfiles should be listed, specified in the format
      `projects|organizations/*/locations/{location}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)