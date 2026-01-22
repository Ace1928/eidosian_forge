from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3TextInput(_messages.Message):
    """Represents the natural language text to be processed.

  Fields:
    text: Required. The UTF-8 encoded natural language text to be processed.
  """
    text = _messages.StringField(1)