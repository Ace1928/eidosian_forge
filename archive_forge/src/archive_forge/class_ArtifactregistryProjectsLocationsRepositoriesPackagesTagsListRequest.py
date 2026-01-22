from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesPackagesTagsListRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesPackagesTagsListRequest
  object.

  Fields:
    filter: An expression for filtering the results of the request. Filter
      rules are case insensitive. The fields eligible for filtering are: *
      `version` An example of using a filter: *
      `version="projects/p1/locations/us-
      central1/repositories/repo1/packages/pkg1/versions/1.0"` --> Tags that
      are applied to the version `1.0` in package `pkg1`.
    pageSize: The maximum number of tags to return. Maximum page size is
      10,000.
    pageToken: The next_page_token value returned from a previous list
      request, if any.
    parent: The name of the parent package whose tags will be listed. For
      example: `projects/p1/locations/us-
      central1/repositories/repo1/packages/pkg1`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)