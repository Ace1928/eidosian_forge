from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTextSegment(_messages.Message):
    """The text segment inside of DataItem.

  Fields:
    content: The text content in the segment for output only.
    endOffset: Zero-based character index of the first character past the end
      of the text segment (counting character from the beginning of the text).
      The character at the end_offset is NOT included in the text segment.
    startOffset: Zero-based character index of the first character of the text
      segment (counting characters from the beginning of the text).
  """
    content = _messages.StringField(1)
    endOffset = _messages.IntegerField(2, variant=_messages.Variant.UINT64)
    startOffset = _messages.IntegerField(3, variant=_messages.Variant.UINT64)