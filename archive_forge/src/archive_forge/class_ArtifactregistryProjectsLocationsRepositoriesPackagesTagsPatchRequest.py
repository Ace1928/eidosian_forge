from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesPackagesTagsPatchRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesPackagesTagsPatchRequest
  object.

  Fields:
    name: The name of the tag, for example: "projects/p1/locations/us-
      central1/repositories/repo1/packages/pkg1/tags/tag1". If the package
      part contains slashes, the slashes are escaped. The tag part can only
      have characters in [a-zA-Z0-9\\-._~:@], anything else must be URL
      encoded.
    tag: A Tag resource to be passed as the request body.
    updateMask: The update mask applies to the resource. For the `FieldMask`
      definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask
  """
    name = _messages.StringField(1, required=True)
    tag = _messages.MessageField('Tag', 2)
    updateMask = _messages.StringField(3)