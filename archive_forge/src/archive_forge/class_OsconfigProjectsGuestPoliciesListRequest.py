from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsGuestPoliciesListRequest(_messages.Message):
    """A OsconfigProjectsGuestPoliciesListRequest object.

  Fields:
    pageSize: The maximum number of guest policies to return.
    pageToken: A pagination token returned from a previous call to
      `ListGuestPolicies` that indicates where this listing should continue
      from.
    parent: Required. The resource name of the parent using one of the
      following forms: `projects/{project_number}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)