from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesAssetsActionsTriggerRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesAssetsActionsTriggerRequest
  object.

  Fields:
    name: Required. The name of the action to Trigger. Format: `projects/{proj
      ect}/locations/{location}/assetTypes/{type}/assets/{asset}/actions/{acti
      on}`
    triggerActionRequest: A TriggerActionRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    triggerActionRequest = _messages.MessageField('TriggerActionRequest', 2)