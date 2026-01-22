from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsTranslateDocumentRequest(_messages.Message):
    """A TranslateProjectsLocationsTranslateDocumentRequest object.

  Fields:
    parent: Required. Location to make a regional call. Format:
      `projects/{project-number-or-id}/locations/{location-id}`. For global
      calls, use `projects/{project-number-or-id}/locations/global`. Non-
      global location is required for requests using AutoML models or custom
      glossaries. Models and glossaries must be within the same region (have
      the same location-id), otherwise an INVALID_ARGUMENT (400) error is
      returned.
    translateDocumentRequest: A TranslateDocumentRequest resource to be passed
      as the request body.
  """
    parent = _messages.StringField(1, required=True)
    translateDocumentRequest = _messages.MessageField('TranslateDocumentRequest', 2)