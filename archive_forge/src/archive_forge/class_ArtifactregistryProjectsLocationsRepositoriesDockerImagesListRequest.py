from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesDockerImagesListRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesDockerImagesListRequest
  object.

  Fields:
    orderBy: The field to order the results by.
    pageSize: The maximum number of artifacts to return.
    pageToken: The next_page_token value returned from a previous list
      request, if any.
    parent: Required. The name of the parent resource whose docker images will
      be listed.
  """
    orderBy = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)