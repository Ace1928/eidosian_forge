from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresContextsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresContextsDeleteRequest object.

  Fields:
    etag: Optional. The etag of the Context to delete. If this is provided, it
      must match the server's etag. Otherwise, the request will fail with a
      FAILED_PRECONDITION.
    force: The force deletion semantics is still undefined. Users should not
      use this field.
    name: Required. The resource name of the Context to delete. Format: `proje
      cts/{project}/locations/{location}/metadataStores/{metadatastore}/contex
      ts/{context}`
  """
    etag = _messages.StringField(1)
    force = _messages.BooleanField(2)
    name = _messages.StringField(3, required=True)