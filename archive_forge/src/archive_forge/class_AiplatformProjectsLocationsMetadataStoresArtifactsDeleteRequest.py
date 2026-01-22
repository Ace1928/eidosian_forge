from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresArtifactsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresArtifactsDeleteRequest
  object.

  Fields:
    etag: Optional. The etag of the Artifact to delete. If this is provided,
      it must match the server's etag. Otherwise, the request will fail with a
      FAILED_PRECONDITION.
    name: Required. The resource name of the Artifact to delete. Format: `proj
      ects/{project}/locations/{location}/metadataStores/{metadatastore}/artif
      acts/{artifact}`
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)