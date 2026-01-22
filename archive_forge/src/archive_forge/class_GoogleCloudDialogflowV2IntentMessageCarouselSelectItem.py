from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageCarouselSelectItem(_messages.Message):
    """An item in the carousel.

  Fields:
    description: Optional. The body text of the card.
    image: Optional. The image to display.
    info: Required. Additional info about the option item.
    title: Required. Title of the carousel item.
  """
    description = _messages.StringField(1)
    image = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageImage', 2)
    info = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageSelectItemInfo', 3)
    title = _messages.StringField(4)