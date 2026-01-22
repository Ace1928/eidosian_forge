from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresContextsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresContextsGetRequest object.

  Fields:
    name: Required. The resource name of the Context to retrieve. Format: `pro
      jects/{project}/locations/{location}/metadataStores/{metadatastore}/cont
      exts/{context}`
  """
    name = _messages.StringField(1, required=True)