from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsconfigProjectsLocationsOsPolicyAssignmentsListRevisionsRequest(_messages.Message):
    """A OsconfigProjectsLocationsOsPolicyAssignmentsListRevisionsRequest
  object.

  Fields:
    name: Required. The name of the OS policy assignment to list revisions
      for.
    pageSize: The maximum number of revisions to return.
    pageToken: A pagination token returned from a previous call to
      `ListOSPolicyAssignmentRevisions` that indicates where this listing
      should continue from.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)