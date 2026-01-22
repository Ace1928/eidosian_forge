from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsconfigProjectsLocationsOsPolicyAssignmentsListRequest(_messages.Message):
    """A OsconfigProjectsLocationsOsPolicyAssignmentsListRequest object.

  Fields:
    pageSize: The maximum number of assignments to return.
    pageToken: A pagination token returned from a previous call to
      `ListOSPolicyAssignments` that indicates where this listing should
      continue from.
    parent: Required. The parent resource name.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)