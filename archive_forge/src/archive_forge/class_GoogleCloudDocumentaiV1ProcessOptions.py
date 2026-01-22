from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1ProcessOptions(_messages.Message):
    """Options for Process API

  Fields:
    fromEnd: Only process certain pages from the end, same as above.
    fromStart: Only process certain pages from the start. Process all if the
      document has fewer pages.
    individualPageSelector: Which pages to process (1-indexed).
    ocrConfig: Only applicable to `OCR_PROCESSOR` and `FORM_PARSER_PROCESSOR`.
      Returns error if set on other processor types.
    schemaOverride: Optional. Override the schema of the ProcessorVersion.
      Will return an Invalid Argument error if this field is set when the
      underlying ProcessorVersion doesn't support schema override.
  """
    fromEnd = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    fromStart = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    individualPageSelector = _messages.MessageField('GoogleCloudDocumentaiV1ProcessOptionsIndividualPageSelector', 3)
    ocrConfig = _messages.MessageField('GoogleCloudDocumentaiV1OcrConfig', 4)
    schemaOverride = _messages.MessageField('GoogleCloudDocumentaiV1DocumentSchema', 5)