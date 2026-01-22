from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchRecognizeFileMetadata(_messages.Message):
    """Metadata about a single file in a batch for BatchRecognize.

  Fields:
    config: Features and audio metadata to use for the Automatic Speech
      Recognition. This field in combination with the config_mask field can be
      used to override parts of the default_recognition_config of the
      Recognizer resource as well as the config at the request level.
    configMask: The list of fields in config that override the values in the
      default_recognition_config of the recognizer during this recognition
      request. If no mask is provided, all non-default valued fields in config
      override the values in the recognizer for this recognition request. If a
      mask is provided, only the fields listed in the mask override the config
      in the recognizer for this recognition request. If a wildcard (`*`) is
      provided, config completely overrides and replaces the config in the
      recognizer for this recognition request.
    uri: Cloud Storage URI for the audio file.
  """
    config = _messages.MessageField('RecognitionConfig', 1)
    configMask = _messages.StringField(2)
    uri = _messages.StringField(3)