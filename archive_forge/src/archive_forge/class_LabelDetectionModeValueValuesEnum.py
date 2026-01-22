from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LabelDetectionModeValueValuesEnum(_messages.Enum):
    """What labels should be detected with LABEL_DETECTION, in addition to
    video-level labels or segment-level labels. If unspecified, defaults to
    `SHOT_MODE`.

    Values:
      LABEL_DETECTION_MODE_UNSPECIFIED: Unspecified.
      SHOT_MODE: Detect shot-level labels.
      FRAME_MODE: Detect frame-level labels.
      SHOT_AND_FRAME_MODE: Detect both shot-level and frame-level labels.
    """
    LABEL_DETECTION_MODE_UNSPECIFIED = 0
    SHOT_MODE = 1
    FRAME_MODE = 2
    SHOT_AND_FRAME_MODE = 3