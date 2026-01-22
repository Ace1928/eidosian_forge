from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p4beta1TextAnnotationDetectedLanguage(_messages.Message):
    """Detected language for a structural component.

  Fields:
    confidence: Confidence of detected language. Range [0, 1].
    languageCode: The BCP-47 language code, such as "en-US" or "sr-Latn". For
      more information, see
      http://www.unicode.org/reports/tr35/#Unicode_locale_identifier.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    languageCode = _messages.StringField(2)