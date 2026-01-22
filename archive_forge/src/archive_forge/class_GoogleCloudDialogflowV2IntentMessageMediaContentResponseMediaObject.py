from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageMediaContentResponseMediaObject(_messages.Message):
    """Response media object for media content card.

  Fields:
    contentUrl: Required. Url where the media is stored.
    description: Optional. Description of media card.
    icon: Optional. Icon to display above media content.
    largeImage: Optional. Image to display above media content.
    name: Required. Name of media card.
  """
    contentUrl = _messages.StringField(1)
    description = _messages.StringField(2)
    icon = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageImage', 3)
    largeImage = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageImage', 4)
    name = _messages.StringField(5)