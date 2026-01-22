from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesPackagesPatchRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesPackagesPatchRequest
  object.

  Fields:
    name: The name of the package, for example: `projects/p1/locations/us-
      central1/repositories/repo1/packages/pkg1`. If the package ID part
      contains slashes, the slashes are escaped.
    package: A Package resource to be passed as the request body.
    updateMask: The update mask applies to the resource. For the `FieldMask`
      definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask
  """
    name = _messages.StringField(1, required=True)
    package = _messages.MessageField('Package', 2)
    updateMask = _messages.StringField(3)