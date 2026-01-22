from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesCreateRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesCreateRequest object.

  Fields:
    parent: Required. The name of the parent resource where the repository
      will be created.
    repository: A Repository resource to be passed as the request body.
    repositoryId: Required. The repository id to use for this repository.
  """
    parent = _messages.StringField(1, required=True)
    repository = _messages.MessageField('Repository', 2)
    repositoryId = _messages.StringField(3)