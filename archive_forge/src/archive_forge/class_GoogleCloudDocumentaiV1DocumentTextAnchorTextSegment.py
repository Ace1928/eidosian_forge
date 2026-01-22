from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentTextAnchorTextSegment(_messages.Message):
    """A text segment in the Document.text. The indices may be out of bounds
  which indicate that the text extends into another document shard for large
  sharded documents. See ShardInfo.text_offset

  Fields:
    endIndex: TextSegment half open end UTF-8 char index in the Document.text.
    startIndex: TextSegment start UTF-8 char index in the Document.text.
  """
    endIndex = _messages.IntegerField(1)
    startIndex = _messages.IntegerField(2)