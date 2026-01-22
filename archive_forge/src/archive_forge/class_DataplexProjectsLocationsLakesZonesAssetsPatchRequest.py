from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesZonesAssetsPatchRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesZonesAssetsPatchRequest object.

  Fields:
    googleCloudDataplexV1Asset: A GoogleCloudDataplexV1Asset resource to be
      passed as the request body.
    name: Output only. The relative resource name of the asset, of the form: p
      rojects/{project_number}/locations/{location_id}/lakes/{lake_id}/zones/{
      zone_id}/assets/{asset_id}.
    updateMask: Required. Mask of fields to update.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    googleCloudDataplexV1Asset = _messages.MessageField('GoogleCloudDataplexV1Asset', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)