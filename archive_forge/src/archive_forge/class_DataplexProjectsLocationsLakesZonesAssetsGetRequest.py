from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesZonesAssetsGetRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesZonesAssetsGetRequest object.

  Fields:
    name: Required. The resource name of the asset: projects/{project_number}/
      locations/{location_id}/lakes/{lake_id}/zones/{zone_id}/assets/{asset_id
      }.
  """
    name = _messages.StringField(1, required=True)