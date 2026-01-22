from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsDetectLanguageRequest(_messages.Message):
    """A TranslateProjectsDetectLanguageRequest object.

  Fields:
    detectLanguageRequest: A DetectLanguageRequest resource to be passed as
      the request body.
    parent: Required. Project or location to make a call. Must refer to a
      caller's project. Format: `projects/{project-number-or-
      id}/locations/{location-id}` or `projects/{project-number-or-id}`. For
      global calls, use `projects/{project-number-or-id}/locations/global` or
      `projects/{project-number-or-id}`. Only models within the same region
      (has same location-id) can be used. Otherwise an INVALID_ARGUMENT (400)
      error is returned.
  """
    detectLanguageRequest = _messages.MessageField('DetectLanguageRequest', 1)
    parent = _messages.StringField(2, required=True)