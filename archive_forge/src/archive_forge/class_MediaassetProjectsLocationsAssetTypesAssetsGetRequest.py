from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesAssetsGetRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesAssetsGetRequest object.

  Fields:
    name: Required. The name of the asset to retrieve, in the following form:
      `projects/{project}/locations/{location}/assetTypes/{type}/assets/{asset
      }`.
    readMask: Extra fields to be poplulated as part of the asset resource in
      the response. Currently, this only supports populating asset metadata
      (no wildcards and no contents of the entire asset).
  """
    name = _messages.StringField(1, required=True)
    readMask = _messages.StringField(2)