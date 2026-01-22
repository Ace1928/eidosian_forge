from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaVideoClassificationAnnotation(_messages.Message):
    """Annotation details specific to video classification.

  Fields:
    annotationSpecId: The resource Id of the AnnotationSpec that this
      Annotation pertains to.
    displayName: The display name of the AnnotationSpec that this Annotation
      pertains to.
    timeSegment: This Annotation applies to the time period represented by the
      TimeSegment. If it's not set, the Annotation applies to the whole video.
  """
    annotationSpecId = _messages.StringField(1)
    displayName = _messages.StringField(2)
    timeSegment = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTimeSegment', 3)