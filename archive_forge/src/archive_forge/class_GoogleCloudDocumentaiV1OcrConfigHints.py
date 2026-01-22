from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1OcrConfigHints(_messages.Message):
    """Hints for OCR Engine

  Fields:
    languageHints: List of BCP-47 language codes to use for OCR. In most
      cases, not specifying it yields the best results since it enables
      automatic language detection. For languages based on the Latin alphabet,
      setting hints is not needed. In rare cases, when the language of the
      text in the image is known, setting a hint will help get better results
      (although it will be a significant hindrance if the hint is wrong).
  """
    languageHints = _messages.StringField(1, repeated=True)