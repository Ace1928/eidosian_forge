from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageContext(_messages.Message):
    """Image context and/or feature-specific parameters.

  Fields:
    cropHintsParams: Parameters for crop hints annotation request.
    languageHints: List of languages to use for TEXT_DETECTION. In most cases,
      an empty value yields the best results since it enables automatic
      language detection. For languages based on the Latin alphabet, setting
      `language_hints` is not needed. In rare cases, when the language of the
      text in the image is known, setting a hint will help get better results
      (although it will be a significant hindrance if the hint is wrong). Text
      detection returns an error if one or more of the specified languages is
      not one of the [supported
      languages](https://cloud.google.com/vision/docs/languages).
    latLongRect: Not used.
    productSearchParams: Parameters for product search.
    textDetectionParams: Parameters for text detection and document text
      detection.
    webDetectionParams: Parameters for web detection.
  """
    cropHintsParams = _messages.MessageField('CropHintsParams', 1)
    languageHints = _messages.StringField(2, repeated=True)
    latLongRect = _messages.MessageField('LatLongRect', 3)
    productSearchParams = _messages.MessageField('ProductSearchParams', 4)
    textDetectionParams = _messages.MessageField('TextDetectionParams', 5)
    webDetectionParams = _messages.MessageField('WebDetectionParams', 6)