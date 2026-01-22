from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictInstanceTextExtractionPredictionInstance(_messages.Message):
    """Prediction input format for Text Extraction.

  Fields:
    content: The text snippet to make the predictions on.
    key: This field is only used for batch prediction. If a key is provided,
      the batch prediction result will by mapped to this key. If omitted, then
      the batch prediction result will contain the entire input instance.
      Vertex AI will not check if keys in the request are duplicates, so it is
      up to the caller to ensure the keys are unique.
    mimeType: The MIME type of the text snippet. The supported MIME types are
      listed below. - text/plain
  """
    content = _messages.StringField(1)
    key = _messages.StringField(2)
    mimeType = _messages.StringField(3)