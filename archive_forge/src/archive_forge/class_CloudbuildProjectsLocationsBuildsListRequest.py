from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsBuildsListRequest(_messages.Message):
    """A CloudbuildProjectsLocationsBuildsListRequest object.

  Fields:
    filter: The raw filter text to constrain the results.
    pageSize: Number of results to return in the list.
    pageToken: The page token for the next page of Builds. If unspecified, the
      first page of results is returned. If the token is rejected for any
      reason, INVALID_ARGUMENT will be thrown. In this case, the token should
      be discarded, and pagination should be restarted from the first page of
      results. See https://google.aip.dev/158 for more.
    parent: The parent of the collection of `Builds`. Format:
      `projects/{project}/locations/{location}`
    projectId: Required. ID of the project.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    projectId = _messages.StringField(5)