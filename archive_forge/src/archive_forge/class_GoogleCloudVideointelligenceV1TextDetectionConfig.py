from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1TextDetectionConfig(_messages.Message):
    """Config for TEXT_DETECTION.

  Fields:
    languageHints: Language hint can be specified if the language to be
      detected is known a priori. It can increase the accuracy of the
      detection. Language hint must be language code in BCP-47 format.
      Automatic language detection is performed if no hint is provided.
    model: Model to use for text detection. Supported values: "builtin/stable"
      (the default if unset) and "builtin/latest".
  """
    languageHints = _messages.StringField(1, repeated=True)
    model = _messages.StringField(2)