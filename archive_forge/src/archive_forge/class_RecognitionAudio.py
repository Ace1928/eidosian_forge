from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecognitionAudio(_messages.Message):
    """Contains audio data in the encoding specified in the
  `RecognitionConfig`. Either `content` or `uri` must be supplied. Supplying
  both or neither returns google.rpc.Code.INVALID_ARGUMENT. See [content
  limits](https://cloud.google.com/speech-to-text/quotas#content).

  Fields:
    content: The audio data bytes encoded as specified in `RecognitionConfig`.
      Note: as with all bytes fields, proto buffers use a pure binary
      representation, whereas JSON representations use base64.
    uri: URI that points to a file that contains audio data bytes as specified
      in `RecognitionConfig`. The file must not be compressed (for example,
      gzip). Currently, only Google Cloud Storage URIs are supported, which
      must be specified in the following format:
      `gs://bucket_name/object_name` (other URI formats return
      google.rpc.Code.INVALID_ARGUMENT). For more information, see [Request
      URIs](https://cloud.google.com/storage/docs/reference-uris).
  """
    content = _messages.BytesField(1)
    uri = _messages.StringField(2)