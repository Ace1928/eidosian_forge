from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityProfilesListRevisionsRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityProfilesListRevisionsRequest object.

  Fields:
    name: Required. For a specific profile, list all the revisions. Format:
      `organizations/{org}/securityProfiles/{profile}`
    pageSize: The maximum number of profile revisions to return. The service
      may return fewer than this value. If unspecified, at most 50 revisions
      will be returned.
    pageToken: A page token, received from a previous
      `ListSecurityProfileRevisions` call. Provide this to retrieve the
      subsequent page.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)