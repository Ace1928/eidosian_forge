from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1FaceDetectionConfig(_messages.Message):
    """Config for FACE_DETECTION.

  Fields:
    includeAttributes: Whether to enable face attributes detection, such as
      glasses, dark_glasses, mouth_open etc. Ignored if
      'include_bounding_boxes' is set to false.
    includeBoundingBoxes: Whether bounding boxes are included in the face
      annotation output.
    model: Model to use for face detection. Supported values: "builtin/stable"
      (the default if unset) and "builtin/latest".
  """
    includeAttributes = _messages.BooleanField(1)
    includeBoundingBoxes = _messages.BooleanField(2)
    model = _messages.StringField(3)