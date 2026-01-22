from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsImageVersionsListRequest(_messages.Message):
    """A ComposerProjectsLocationsImageVersionsListRequest object.

  Fields:
    includePastReleases: Whether or not image versions from old releases
      should be included.
    pageSize: The maximum number of image_versions to return.
    pageToken: The next_page_token value returned from a previous List
      request, if any.
    parent: List ImageVersions in the given project and location, in the form:
      "projects/{projectId}/locations/{locationId}"
  """
    includePastReleases = _messages.BooleanField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)