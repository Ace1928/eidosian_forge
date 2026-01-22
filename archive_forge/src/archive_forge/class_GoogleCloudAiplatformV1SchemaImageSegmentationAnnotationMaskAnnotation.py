from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaImageSegmentationAnnotationMaskAnnotation(_messages.Message):
    """The mask based segmentation annotation.

  Fields:
    annotationSpecColors: The mapping between color and AnnotationSpec for
      this Annotation.
    maskGcsUri: Google Cloud Storage URI that points to the mask image. The
      image must be in PNG format. It must have the same size as the
      DataItem's image. Each pixel in the image mask represents the
      AnnotationSpec which the pixel in the image DataItem belong to. Each
      color is mapped to one AnnotationSpec based on annotation_spec_colors.
  """
    annotationSpecColors = _messages.MessageField('GoogleCloudAiplatformV1SchemaAnnotationSpecColor', 1, repeated=True)
    maskGcsUri = _messages.StringField(2)