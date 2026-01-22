from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesFilesListRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesFilesListRequest object.

  Fields:
    filter: An expression for filtering the results of the request. Filter
      rules are case insensitive. The fields eligible for filtering are: *
      `name` * `owner` An example of using a filter: *
      `name="projects/p1/locations/us-
      central1/repositories/repo1/files/a/b/*"` --> Files with an ID starting
      with "a/b/". * `owner="projects/p1/locations/us-
      central1/repositories/repo1/packages/pkg1/versions/1.0"` --> Files owned
      by the version `1.0` in package `pkg1`.
    orderBy: The field to order the results by.
    pageSize: The maximum number of files to return.
    pageToken: The next_page_token value returned from a previous list
      request, if any.
    parent: Required. The name of the repository whose files will be listed.
      For example: "projects/p1/locations/us-central1/repositories/repo1
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)