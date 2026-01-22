from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesRulesListRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesRulesListRequest object.

  Fields:
    pageSize: The maximum number of rules to return. Maximum page size is
      10,000.
    pageToken: The next_page_token value returned from a previous list
      request, if any.
    parent: Required. The name of the parent repository whose rules will be
      listed. For example: `projects/p1/locations/us-
      central1/repositories/repo1`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)