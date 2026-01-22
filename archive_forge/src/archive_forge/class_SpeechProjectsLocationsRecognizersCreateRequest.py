from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechProjectsLocationsRecognizersCreateRequest(_messages.Message):
    """A SpeechProjectsLocationsRecognizersCreateRequest object.

  Fields:
    parent: Required. The project and location where this Recognizer will be
      created. The expected format is
      `projects/{project}/locations/{location}`.
    recognizer: A Recognizer resource to be passed as the request body.
    recognizerId: The ID to use for the Recognizer, which will become the
      final component of the Recognizer's resource name. This value should be
      4-63 characters, and valid characters are /a-z-/.
    validateOnly: If set, validate the request and preview the Recognizer, but
      do not actually create it.
  """
    parent = _messages.StringField(1, required=True)
    recognizer = _messages.MessageField('Recognizer', 2)
    recognizerId = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)