from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechProjectsLocationsCustomClassesGetRequest(_messages.Message):
    """A SpeechProjectsLocationsCustomClassesGetRequest object.

  Fields:
    name: Required. The name of the CustomClass to retrieve. The expected
      format is
      `projects/{project}/locations/{location}/customClasses/{custom_class}`.
  """
    name = _messages.StringField(1, required=True)