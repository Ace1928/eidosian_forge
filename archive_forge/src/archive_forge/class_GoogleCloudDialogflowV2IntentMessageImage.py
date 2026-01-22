from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageImage(_messages.Message):
    """The image response message.

  Fields:
    accessibilityText: Optional. A text description of the image to be used
      for accessibility, e.g., screen readers.
    imageUri: Optional. The public URI to an image file.
  """
    accessibilityText = _messages.StringField(1)
    imageUri = _messages.StringField(2)