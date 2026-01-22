from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechProjectsLocationsConfigUpdateRequest(_messages.Message):
    """A SpeechProjectsLocationsConfigUpdateRequest object.

  Fields:
    config: A Config resource to be passed as the request body.
    name: Output only. Identifier. The name of the config resource. There is
      exactly one config resource per project per location. The expected
      format is `projects/{project}/locations/{location}/config`.
    updateMask: The list of fields to be updated.
  """
    config = _messages.MessageField('Config', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)