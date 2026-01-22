from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AudioStream(_messages.Message):
    """Audio stream resource.

  Fields:
    bitrateBps: Required. Audio bitrate in bits per second. Must be between 1
      and 10,000,000.
    channelCount: Number of audio channels. Must be between 1 and 6. The
      default is 2.
    channelLayout: A list of channel names specifying layout of the audio
      channels. This only affects the metadata embedded in the container
      headers, if supported by the specified format. The default is `["fl",
      "fr"]`. Supported channel names: - `fl` - Front left channel - `fr` -
      Front right channel - `sl` - Side left channel - `sr` - Side right
      channel - `fc` - Front center channel - `lfe` - Low frequency
    codec: The codec for this audio stream. The default is `aac`. Supported
      audio codecs: - `aac` - `aac-he` - `aac-he-v2` - `mp3` - `ac3` - `eac3`
    displayName: The name for this particular audio stream that will be added
      to the HLS/DASH manifest. Not supported in MP4 files.
    languageCode: The BCP-47 language code, such as `en-US` or `sr-Latn`. For
      more information, see
      https://www.unicode.org/reports/tr35/#Unicode_locale_identifier. Not
      supported in MP4 files.
    mapping: The mapping for the JobConfig.edit_list atoms with audio
      EditAtom.inputs.
    sampleRateHertz: The audio sample rate in Hertz. The default is 48000
      Hertz.
  """
    bitrateBps = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    channelCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    channelLayout = _messages.StringField(3, repeated=True)
    codec = _messages.StringField(4)
    displayName = _messages.StringField(5)
    languageCode = _messages.StringField(6)
    mapping = _messages.MessageField('AudioMapping', 7, repeated=True)
    sampleRateHertz = _messages.IntegerField(8, variant=_messages.Variant.INT32)