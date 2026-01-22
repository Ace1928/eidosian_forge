from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnnotateImageResponse(_messages.Message):
    """Response to an image annotation request.

  Fields:
    context: If present, contextual information is needed to understand where
      this image comes from.
    cropHintsAnnotation: If present, crop hints have completed successfully.
    error: If set, represents the error message for the operation. Note that
      filled-in image annotations are guaranteed to be correct, even when
      `error` is set.
    faceAnnotations: If present, face detection has completed successfully.
    fullTextAnnotation: If present, text (OCR) detection or document (OCR)
      text detection has completed successfully. This annotation provides the
      structural hierarchy for the OCR detected text.
    imagePropertiesAnnotation: If present, image properties were extracted
      successfully.
    labelAnnotations: If present, label detection has completed successfully.
    landmarkAnnotations: If present, landmark detection has completed
      successfully.
    localizedObjectAnnotations: If present, localized object detection has
      completed successfully. This will be sorted descending by confidence
      score.
    logoAnnotations: If present, logo detection has completed successfully.
    productSearchResults: If present, product search has completed
      successfully.
    safeSearchAnnotation: If present, safe-search annotation has completed
      successfully.
    textAnnotations: If present, text (OCR) detection has completed
      successfully.
    webDetection: If present, web detection has completed successfully.
  """
    context = _messages.MessageField('ImageAnnotationContext', 1)
    cropHintsAnnotation = _messages.MessageField('CropHintsAnnotation', 2)
    error = _messages.MessageField('Status', 3)
    faceAnnotations = _messages.MessageField('FaceAnnotation', 4, repeated=True)
    fullTextAnnotation = _messages.MessageField('TextAnnotation', 5)
    imagePropertiesAnnotation = _messages.MessageField('ImageProperties', 6)
    labelAnnotations = _messages.MessageField('EntityAnnotation', 7, repeated=True)
    landmarkAnnotations = _messages.MessageField('EntityAnnotation', 8, repeated=True)
    localizedObjectAnnotations = _messages.MessageField('LocalizedObjectAnnotation', 9, repeated=True)
    logoAnnotations = _messages.MessageField('EntityAnnotation', 10, repeated=True)
    productSearchResults = _messages.MessageField('ProductSearchResults', 11)
    safeSearchAnnotation = _messages.MessageField('SafeSearchAnnotation', 12)
    textAnnotations = _messages.MessageField('EntityAnnotation', 13, repeated=True)
    webDetection = _messages.MessageField('WebDetection', 14)