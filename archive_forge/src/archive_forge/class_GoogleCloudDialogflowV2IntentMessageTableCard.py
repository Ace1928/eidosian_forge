from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageTableCard(_messages.Message):
    """Table card for Actions on Google.

  Fields:
    buttons: Optional. List of buttons for the card.
    columnProperties: Optional. Display properties for the columns in this
      table.
    image: Optional. Image which should be displayed on the card.
    rows: Optional. Rows in this table of data.
    subtitle: Optional. Subtitle to the title.
    title: Required. Title of the card.
  """
    buttons = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageBasicCardButton', 1, repeated=True)
    columnProperties = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageColumnProperties', 2, repeated=True)
    image = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageImage', 3)
    rows = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageTableCardRow', 4, repeated=True)
    subtitle = _messages.StringField(5)
    title = _messages.StringField(6)