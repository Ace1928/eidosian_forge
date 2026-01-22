from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsConnectionsFetchLinkableRepositoriesRequest(_messages.Message):
    """A CloudbuildProjectsLocationsConnectionsFetchLinkableRepositoriesRequest
  object.

  Fields:
    connection: Required. The name of the Connection. Format:
      `projects/*/locations/*/connections/*`.
    pageSize: Number of results to return in the list. Default to 20.
    pageToken: Page start.
  """
    connection = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)