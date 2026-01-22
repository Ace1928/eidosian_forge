from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TextSpan(_messages.Message):
    """Represents a text span in the input document.

  Fields:
    beginOffset: The API calculates the beginning offset of the content in the
      original document according to the EncodingType specified in the API
      request.
    content: The content of the text span, which is a substring of the
      document.
  """
    beginOffset = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    content = _messages.StringField(2)