from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentTextAnchor(_messages.Message):
    """Text reference indexing into the Document.text.

  Fields:
    content: Contains the content of the text span so that users do not have
      to look it up in the text_segments. It is always populated for
      formFields.
    textSegments: The text segments from the Document.text.
  """
    content = _messages.StringField(1)
    textSegments = _messages.MessageField('GoogleCloudDocumentaiV1DocumentTextAnchorTextSegment', 2, repeated=True)