from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesPatchRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesPatchRequest object.

  Fields:
    name: The name of the repository, for example: `projects/p1/locations/us-
      central1/repositories/repo1`.
    repository: A Repository resource to be passed as the request body.
    updateMask: The update mask applies to the resource. For the `FieldMask`
      definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask
  """
    name = _messages.StringField(1, required=True)
    repository = _messages.MessageField('Repository', 2)
    updateMask = _messages.StringField(3)