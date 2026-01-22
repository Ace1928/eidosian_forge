from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesRulesGetRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesRulesGetRequest object.

  Fields:
    name: Required. The name of the Rule to retrieve. Format:
      `projects/{project}/locations/{location}/assetTypes/{type}/rules/{rule}`
  """
    name = _messages.StringField(1, required=True)