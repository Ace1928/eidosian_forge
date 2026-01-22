from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OverlayTypeValueValuesEnum(_messages.Enum):
    """How the original image is displayed in the visualization. Adjusting
    the overlay can help increase visual clarity if the original image makes
    it difficult to view the visualization. Defaults to NONE.

    Values:
      OVERLAY_TYPE_UNSPECIFIED: Default value. This is the same as NONE.
      NONE: No overlay.
      ORIGINAL: The attributions are shown on top of the original image.
      GRAYSCALE: The attributions are shown on top of grayscaled version of
        the original image.
      MASK_BLACK: The attributions are used as a mask to reveal predictive
        parts of the image and hide the un-predictive parts.
    """
    OVERLAY_TYPE_UNSPECIFIED = 0
    NONE = 1
    ORIGINAL = 2
    GRAYSCALE = 3
    MASK_BLACK = 4