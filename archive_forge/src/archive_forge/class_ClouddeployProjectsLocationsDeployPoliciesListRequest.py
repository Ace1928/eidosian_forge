from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeployPoliciesListRequest(_messages.Message):
    """A ClouddeployProjectsLocationsDeployPoliciesListRequest object.

  Fields:
    filter: Filter deploy policies to be returned. See
      https://google.aip.dev/160 for more details. All fields can be used in
      the filter.
    orderBy: Field to sort by. See https://google.aip.dev/132#ordering for
      more details.
    pageSize: The maximum number of deploy policies to return. The service may
      return fewer than this value. If unspecified, at most 50 deploy policies
      will be returned. The maximum value is 1000; values above 1000 will be
      set to 1000.
    pageToken: A page token, received from a previous `ListDeployPolicies`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other provided parameters match the call that provided the page token.
    parent: Required. The parent, which owns this collection of deploy
      policies. Format must be
      `projects/{project_id}/locations/{location_name}`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)