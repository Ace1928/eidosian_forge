from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MuxStream(_messages.Message):
    """Multiplexing settings for output stream.

  Fields:
    container: The container format. The default is `mp4` Supported container
      formats: - `ts` - `fmp4`- the corresponding file extension is `.m4s` -
      `mp4` - `vtt` See also: [Supported input and output
      formats](https://cloud.google.com/transcoder/docs/concepts/supported-
      input-and-output-formats)
    elementaryStreams: List of ElementaryStream.key values multiplexed in this
      stream.
    encryptionId: Identifier of the encryption configuration to use. If
      omitted, output will be unencrypted.
    fileName: The name of the generated file. The default is MuxStream.key
      with the extension suffix corresponding to the MuxStream.container.
      Individual segments also have an incremental 10-digit zero-padded suffix
      starting from 0 before the extension, such as `mux_stream0000000123.ts`.
    fmp4: Optional. `fmp4` container configuration.
    key: A unique key for this multiplexed stream.
    segmentSettings: Segment settings for `ts`, `fmp4` and `vtt`.
  """
    container = _messages.StringField(1)
    elementaryStreams = _messages.StringField(2, repeated=True)
    encryptionId = _messages.StringField(3)
    fileName = _messages.StringField(4)
    fmp4 = _messages.MessageField('Fmp4Config', 5)
    key = _messages.StringField(6)
    segmentSettings = _messages.MessageField('SegmentSettings', 7)