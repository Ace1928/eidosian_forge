from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageCard(_messages.Message):
    """The card response message.

  Fields:
    buttons: Optional. The collection of card buttons.
    imageUri: Optional. The public URI to an image file for the card.
    subtitle: Optional. The subtitle of the card.
    title: Optional. The title of the card.
  """
    buttons = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageCardButton', 1, repeated=True)
    imageUri = _messages.StringField(2)
    subtitle = _messages.StringField(3)
    title = _messages.StringField(4)