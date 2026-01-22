from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechProjectsLocationsConfigGetRequest(_messages.Message):
    """A SpeechProjectsLocationsConfigGetRequest object.

  Fields:
    name: Required. The name of the config to retrieve. There is exactly one
      config resource per project per location. The expected format is
      `projects/{project}/locations/{location}/config`.
  """
    name = _messages.StringField(1, required=True)