from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1VertexModelSpec(_messages.Message):
    """Specification for vertex model resources.

  Fields:
    containerImageUri: URI of the Docker image to be used as the custom
      container for serving predictions.
    versionAliases: User provided version aliases so that a model version can
      be referenced via alias
    versionDescription: The description of this version.
    versionId: The version ID of the model.
    vertexModelSourceInfo: Source of a Vertex model.
  """
    containerImageUri = _messages.StringField(1)
    versionAliases = _messages.StringField(2, repeated=True)
    versionDescription = _messages.StringField(3)
    versionId = _messages.StringField(4)
    vertexModelSourceInfo = _messages.MessageField('GoogleCloudDatacatalogV1VertexModelSourceInfo', 5)