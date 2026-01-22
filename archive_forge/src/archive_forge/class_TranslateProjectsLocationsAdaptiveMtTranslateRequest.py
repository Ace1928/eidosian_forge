from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsAdaptiveMtTranslateRequest(_messages.Message):
    """A TranslateProjectsLocationsAdaptiveMtTranslateRequest object.

  Fields:
    adaptiveMtTranslateRequest: A AdaptiveMtTranslateRequest resource to be
      passed as the request body.
    parent: Required. Location to make a regional call. Format:
      `projects/{project-number-or-id}/locations/{location-id}`.
  """
    adaptiveMtTranslateRequest = _messages.MessageField('AdaptiveMtTranslateRequest', 1)
    parent = _messages.StringField(2, required=True)