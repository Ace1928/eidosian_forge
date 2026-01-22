from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesAssetsActionsCancelRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesAssetsActionsCancelRequest
  object.

  Fields:
    cancelActionRequest: A CancelActionRequest resource to be passed as the
      request body.
    name: Required. The name of the action to cancel. Format: `projects/{proje
      ct}/locations/{location}/assetTypes/{type}/assets/{asset}/actions/{actio
      n}`
  """
    cancelActionRequest = _messages.MessageField('CancelActionRequest', 1)
    name = _messages.StringField(2, required=True)