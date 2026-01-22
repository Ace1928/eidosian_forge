from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTextSentimentAnnotation(_messages.Message):
    """Annotation details specific to text sentiment.

  Fields:
    annotationSpecId: The resource Id of the AnnotationSpec that this
      Annotation pertains to.
    displayName: The display name of the AnnotationSpec that this Annotation
      pertains to.
    sentiment: The sentiment score for text.
    sentimentMax: The sentiment max score for text.
  """
    annotationSpecId = _messages.StringField(1)
    displayName = _messages.StringField(2)
    sentiment = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    sentimentMax = _messages.IntegerField(4, variant=_messages.Variant.INT32)