from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TextStream(_messages.Message):
    """Encoding of a text stream. For example, closed captions or subtitles.

  Fields:
    codec: The codec for this text stream. The default is `webvtt`. Supported
      text codecs: - `srt` - `ttml` - `cea608` - `cea708` - `webvtt`
    displayName: The name for this particular text stream that will be added
      to the HLS/DASH manifest. Not supported in MP4 files.
    languageCode: The BCP-47 language code, such as `en-US` or `sr-Latn`. For
      more information, see
      https://www.unicode.org/reports/tr35/#Unicode_locale_identifier. Not
      supported in MP4 files.
    mapping: The mapping for the JobConfig.edit_list atoms with text
      EditAtom.inputs.
  """
    codec = _messages.StringField(1)
    displayName = _messages.StringField(2)
    languageCode = _messages.StringField(3)
    mapping = _messages.MessageField('TextMapping', 4, repeated=True)