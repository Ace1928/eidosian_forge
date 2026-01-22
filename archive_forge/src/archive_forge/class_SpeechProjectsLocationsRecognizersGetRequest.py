from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechProjectsLocationsRecognizersGetRequest(_messages.Message):
    """A SpeechProjectsLocationsRecognizersGetRequest object.

  Fields:
    name: Required. The name of the Recognizer to retrieve. The expected
      format is
      `projects/{project}/locations/{location}/recognizers/{recognizer}`.
  """
    name = _messages.StringField(1, required=True)