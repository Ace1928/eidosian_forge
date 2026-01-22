from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OutputFormatConfig(_messages.Message):
    """Configuration for the format of the results stored to `output`.

  Fields:
    native: Configuration for the native output format. If this field is set
      or if no other output format field is set then transcripts will be
      written to the sink in the native format.
    srt: Configuration for the srt output format. If this field is set then
      transcripts will be written to the sink in the srt format.
    vtt: Configuration for the vtt output format. If this field is set then
      transcripts will be written to the sink in the vtt format.
  """
    native = _messages.MessageField('NativeOutputFileFormatConfig', 1)
    srt = _messages.MessageField('SrtOutputFileFormatConfig', 2)
    vtt = _messages.MessageField('VttOutputFileFormatConfig', 3)