from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesAptArtifactsUploadRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesAptArtifactsUploadRequest
  object.

  Fields:
    parent: The name of the parent resource where the artifacts will be
      uploaded.
    uploadAptArtifactRequest: A UploadAptArtifactRequest resource to be passed
      as the request body.
  """
    parent = _messages.StringField(1, required=True)
    uploadAptArtifactRequest = _messages.MessageField('UploadAptArtifactRequest', 2)