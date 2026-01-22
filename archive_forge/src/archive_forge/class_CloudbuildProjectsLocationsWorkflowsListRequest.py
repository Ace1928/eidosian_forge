from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsWorkflowsListRequest(_messages.Message):
    """A CloudbuildProjectsLocationsWorkflowsListRequest object.

  Fields:
    filter: Filter for the results.
    orderBy: The order to sort results by.
    pageSize: Number of results to return in the list.
    pageToken: Page start.
    parent: Required. Format: `projects/{project}/locations/{location}`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)