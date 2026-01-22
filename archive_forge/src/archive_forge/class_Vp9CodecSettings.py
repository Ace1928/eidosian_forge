from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Vp9CodecSettings(_messages.Message):
    """VP9 codec settings.

  Enums:
    FrameRateConversionStrategyValueValuesEnum: Optional. Frame rate
      conversion strategy for desired frame rate. The default is `DOWNSAMPLE`.

  Fields:
    bitrateBps: Required. The video bitrate in bits per second. The minimum
      value is 1,000. The maximum value is 480,000,000.
    crfLevel: Target CRF level. Must be between 10 and 36, where 10 is the
      highest quality and 36 is the most efficient compression. The default is
      21. **Note:** This field is not supported.
    frameRate: Required. The target video frame rate in frames per second
      (FPS). Must be less than or equal to 120.
    frameRateConversionStrategy: Optional. Frame rate conversion strategy for
      desired frame rate. The default is `DOWNSAMPLE`.
    gopDuration: Select the GOP size based on the specified duration. The
      default is `3s`. Note that `gopDuration` must be less than or equal to
      [`segmentDuration`](#SegmentSettings), and
      [`segmentDuration`](#SegmentSettings) must be divisible by
      `gopDuration`.
    gopFrameCount: Select the GOP size based on the specified frame count.
      Must be greater than zero.
    heightPixels: The height of the video in pixels. Must be an even integer.
      When not specified, the height is adjusted to match the specified width
      and input aspect ratio. If both are omitted, the input height is used.
      For portrait videos that contain horizontal ASR and rotation metadata,
      provide the height, in pixels, per the horizontal ASR. The API
      calculates the width per the horizontal ASR. The API detects any
      rotation metadata and swaps the requested height and width for the
      output.
    hlg: Optional. HLG color format setting for VP9.
    pixelFormat: Pixel format to use. The default is `yuv420p`. Supported
      pixel formats: - `yuv420p` pixel format - `yuv422p` pixel format -
      `yuv444p` pixel format - `yuv420p10` 10-bit HDR pixel format -
      `yuv422p10` 10-bit HDR pixel format - `yuv444p10` 10-bit HDR pixel
      format - `yuv420p12` 12-bit HDR pixel format - `yuv422p12` 12-bit HDR
      pixel format - `yuv444p12` 12-bit HDR pixel format
    profile: Enforces the specified codec profile. The following profiles are
      supported: * `profile0` (default) * `profile1` * `profile2` * `profile3`
      The available options are [WebM-
      compatible](https://www.webmproject.org/vp9/profiles/). Note that
      certain values for this field may cause the transcoder to override other
      fields you set in the `Vp9CodecSettings` message.
    rateControlMode: Specify the mode. The default is `vbr`. Supported rate
      control modes: - `vbr` - variable bitrate
    sdr: Optional. SDR color format setting for VP9.
    widthPixels: The width of the video in pixels. Must be an even integer.
      When not specified, the width is adjusted to match the specified height
      and input aspect ratio. If both are omitted, the input width is used.
      For portrait videos that contain horizontal ASR and rotation metadata,
      provide the width, in pixels, per the horizontal ASR. The API calculates
      the height per the horizontal ASR. The API detects any rotation metadata
      and swaps the requested height and width for the output.
  """

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
    bitrateBps = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    crfLevel = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    frameRate = _messages.FloatField(3)
    frameRateConversionStrategy = _messages.EnumField('FrameRateConversionStrategyValueValuesEnum', 4)
    gopDuration = _messages.StringField(5)
    gopFrameCount = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    heightPixels = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    hlg = _messages.MessageField('Vp9ColorFormatHLG', 8)
    pixelFormat = _messages.StringField(9)
    profile = _messages.StringField(10)
    rateControlMode = _messages.StringField(11)
    sdr = _messages.MessageField('Vp9ColorFormatSDR', 12)
    widthPixels = _messages.IntegerField(13, variant=_messages.Variant.INT32)