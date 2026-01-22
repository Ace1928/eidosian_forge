from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechProjectsLocationsCustomClassesPatchRequest(_messages.Message):
    """A SpeechProjectsLocationsCustomClassesPatchRequest object.

  Fields:
    customClass: A CustomClass resource to be passed as the request body.
    name: Output only. Identifier. The resource name of the CustomClass.
      Format:
      `projects/{project}/locations/{location}/customClasses/{custom_class}`.
    updateMask: The list of fields to be updated. If empty, all fields are
      considered for update.
    validateOnly: If set, validate the request and preview the updated
      CustomClass, but do not actually update it.
  """
    customClass = _messages.MessageField('CustomClass', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)