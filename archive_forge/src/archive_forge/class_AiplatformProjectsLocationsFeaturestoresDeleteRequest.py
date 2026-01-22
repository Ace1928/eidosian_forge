from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresDeleteRequest object.

  Fields:
    force: If set to true, any EntityTypes and Features for this Featurestore
      will also be deleted. (Otherwise, the request will only work if the
      Featurestore has no EntityTypes.)
    name: Required. The name of the Featurestore to be deleted. Format:
      `projects/{project}/locations/{location}/featurestores/{featurestore}`
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)