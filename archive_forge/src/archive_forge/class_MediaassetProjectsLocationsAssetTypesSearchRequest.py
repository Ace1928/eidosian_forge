from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesSearchRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesSearchRequest object.

  Fields:
    name: Required. The asset type resource name, in the following form:
      `projects/{project}/locations/{location}/assetTypes/{type}`.
    searchAssetTypeRequest: A SearchAssetTypeRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    searchAssetTypeRequest = _messages.MessageField('SearchAssetTypeRequest', 2)