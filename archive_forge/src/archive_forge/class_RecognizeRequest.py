from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecognizeRequest(_messages.Message):
    """Request message for the Recognize method. Either `content` or `uri` must
  be supplied. Supplying both or neither returns INVALID_ARGUMENT. See
  [content limits](https://cloud.google.com/speech-to-text/quotas#content).

  Fields:
    config: Features and audio metadata to use for the Automatic Speech
      Recognition. This field in combination with the config_mask field can be
      used to override parts of the default_recognition_config of the
      Recognizer resource.
    configMask: The list of fields in config that override the values in the
      default_recognition_config of the recognizer during this recognition
      request. If no mask is provided, all non-default valued fields in config
      override the values in the recognizer for this recognition request. If a
      mask is provided, only the fields listed in the mask override the config
      in the recognizer for this recognition request. If a wildcard (`*`) is
      provided, config completely overrides and replaces the config in the
      recognizer for this recognition request.
    content: The audio data bytes encoded as specified in RecognitionConfig.
      As with all bytes fields, proto buffers use a pure binary
      representation, whereas JSON representations use base64.
    uri: URI that points to a file that contains audio data bytes as specified
      in RecognitionConfig. The file must not be compressed (for example,
      gzip). Currently, only Google Cloud Storage URIs are supported, which
      must be specified in the following format:
      `gs://bucket_name/object_name` (other URI formats return
      INVALID_ARGUMENT). For more information, see [Request
      URIs](https://cloud.google.com/storage/docs/reference-uris).
  """
    config = _messages.MessageField('RecognitionConfig', 1)
    configMask = _messages.StringField(2)
    content = _messages.BytesField(3)
    uri = _messages.StringField(4)