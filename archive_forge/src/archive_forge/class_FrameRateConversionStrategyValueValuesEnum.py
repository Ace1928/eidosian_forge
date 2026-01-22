from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FrameRateConversionStrategyValueValuesEnum(_messages.Enum):
    """Optional. Frame rate conversion strategy for desired frame rate. The
    default is `DOWNSAMPLE`.

    Values:
      FRAME_RATE_CONVERSION_STRATEGY_UNSPECIFIED: Unspecified frame rate
        conversion strategy.
      DOWNSAMPLE: Selectively retain frames to reduce the output frame rate.
        Every _n_ th frame is kept, where `n = ceil(input frame rate / target
        frame rate)`. When _n_ = 1 (that is, the target frame rate is greater
        than the input frame rate), the output frame rate matches the input
        frame rate. When _n_ > 1, frames are dropped and the output frame rate
        is equal to `(input frame rate / n)`. For more information, see
        [Calculate frame
        rate](https://cloud.google.com/transcoder/docs/concepts/frame-rate).
      DROP_DUPLICATE: Drop or duplicate frames to match the specified frame
        rate.
    """
    FRAME_RATE_CONVERSION_STRATEGY_UNSPECIFIED = 0
    DOWNSAMPLE = 1
    DROP_DUPLICATE = 2