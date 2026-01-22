from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesPackagesTagsDeleteRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesPackagesTagsDeleteRequest
  object.

  Fields:
    name: The name of the tag to delete.
  """
    name = _messages.StringField(1, required=True)