from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecognitionOutputConfig(_messages.Message):
    """Configuration options for the output(s) of recognition.

  Fields:
    gcsOutputConfig: If this message is populated, recognition results are
      written to the provided Google Cloud Storage URI.
    inlineResponseConfig: If this message is populated, recognition results
      are provided in the BatchRecognizeResponse message of the Operation when
      completed. This is only supported when calling BatchRecognize with just
      one audio file.
    outputFormatConfig: Optional. Configuration for the format of the results
      stored to `output`. If unspecified transcripts will be written in the
      `NATIVE` format only.
  """
    gcsOutputConfig = _messages.MessageField('GcsOutputConfig', 1)
    inlineResponseConfig = _messages.MessageField('InlineOutputConfig', 2)
    outputFormatConfig = _messages.MessageField('OutputFormatConfig', 3)