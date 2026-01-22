from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1PersonDetectionConfig(_messages.Message):
    """Config for PERSON_DETECTION.

  Fields:
    includeAttributes: Whether to enable person attributes detection, such as
      cloth color (black, blue, etc), type (coat, dress, etc), pattern (plain,
      floral, etc), hair, etc. Ignored if 'include_bounding_boxes' is set to
      false.
    includeBoundingBoxes: Whether bounding boxes are included in the person
      detection annotation output.
    includePoseLandmarks: Whether to enable pose landmarks detection. Ignored
      if 'include_bounding_boxes' is set to false.
  """
    includeAttributes = _messages.BooleanField(1)
    includeBoundingBoxes = _messages.BooleanField(2)
    includePoseLandmarks = _messages.BooleanField(3)