from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesGenericArtifactsUploadRequest(_messages.Message):
    """A
  ArtifactregistryProjectsLocationsRepositoriesGenericArtifactsUploadRequest
  object.

  Fields:
    parent: The resource name of the repository where the generic artifact
      will be uploaded.
    uploadGenericArtifactRequest: A UploadGenericArtifactRequest resource to
      be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    uploadGenericArtifactRequest = _messages.MessageField('UploadGenericArtifactRequest', 2)