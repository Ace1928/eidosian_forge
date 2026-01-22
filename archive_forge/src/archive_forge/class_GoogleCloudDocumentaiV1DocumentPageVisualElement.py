from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentPageVisualElement(_messages.Message):
    """Detected non-text visual elements e.g. checkbox, signature etc. on the
  page.

  Fields:
    detectedLanguages: A list of detected languages together with confidence.
    layout: Layout for VisualElement.
    type: Type of the VisualElement.
  """
    detectedLanguages = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageDetectedLanguage', 1, repeated=True)
    layout = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageLayout', 2)
    type = _messages.StringField(3)