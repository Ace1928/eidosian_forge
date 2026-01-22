from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1AssetMoveAnalysis(_messages.Message):
    """Represents move analysis results for an asset.

  Fields:
    analysisGroups: List of eligible analyses performed for the asset.
    asset: The full resource name of the asset being analyzed. Example: //comp
      ute.googleapis.com/projects/my_project_123/zones/zone1/instances/instanc
      e1
    assetType: Type of the asset being analyzed. Possible values will be among
      the ones listed [here](https://cloud.google.com/asset-
      inventory/docs/supported-asset-types).
  """
    analysisGroups = _messages.MessageField('GoogleCloudAssuredworkloadsV1MoveAnalysisGroup', 1, repeated=True)
    asset = _messages.StringField(2)
    assetType = _messages.StringField(3)