from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SoftwareRecipeArtifact(_messages.Message):
    """Specifies a resource to be used in the recipe.

  Fields:
    allowInsecure: Defaults to false. When false, recipes are subject to
      validations based on the artifact type: Remote: A checksum must be
      specified, and only protocols with transport-layer security are
      permitted. GCS: An object generation number must be specified.
    gcs: A Google Cloud Storage artifact.
    id: Required. Id of the artifact, which the installation and update steps
      of this recipe can reference. Artifacts in a recipe cannot have the same
      id.
    remote: A generic remote artifact.
  """
    allowInsecure = _messages.BooleanField(1)
    gcs = _messages.MessageField('SoftwareRecipeArtifactGcs', 2)
    id = _messages.StringField(3)
    remote = _messages.MessageField('SoftwareRecipeArtifactRemote', 4)