from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1DeployedIndexRef(_messages.Message):
    """Points to a DeployedIndex.

  Fields:
    deployedIndexId: Immutable. The ID of the DeployedIndex in the above
      IndexEndpoint.
    displayName: Output only. The display name of the DeployedIndex.
    indexEndpoint: Immutable. A resource name of the IndexEndpoint.
  """
    deployedIndexId = _messages.StringField(1)
    displayName = _messages.StringField(2)
    indexEndpoint = _messages.StringField(3)