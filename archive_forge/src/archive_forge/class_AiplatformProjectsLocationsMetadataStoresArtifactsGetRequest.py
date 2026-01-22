from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresArtifactsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresArtifactsGetRequest object.

  Fields:
    name: Required. The resource name of the Artifact to retrieve. Format: `pr
      ojects/{project}/locations/{location}/metadataStores/{metadatastore}/art
      ifacts/{artifact}`
  """
    name = _messages.StringField(1, required=True)