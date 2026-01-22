from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsListRequest(_messages.Message):
    """A
  PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsListRequest
  object.

  Fields:
    pageSize: Optional. The maximum number of items to return. The service may
      return fewer than this value. If unspecified, at most 5 items will be
      returned. The maximum value is 10; values above 10 will be coerced to
      10.
    pageToken: Optional. A page token, received from a previous call. Provide
      this to retrieve the subsequent page. When paginating, all other
      parameters must match the call that provided the page token.
    parent: Required. The parent the violations are scoped to. Format:
      `organizations/{organization}/locations/{location}` Example:
      `organizations/my-example-org/locations/global`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)