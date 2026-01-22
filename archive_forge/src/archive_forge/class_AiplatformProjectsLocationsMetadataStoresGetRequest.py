from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresGetRequest object.

  Fields:
    name: Required. The resource name of the MetadataStore to retrieve.
      Format:
      `projects/{project}/locations/{location}/metadataStores/{metadatastore}`
  """
    name = _messages.StringField(1, required=True)